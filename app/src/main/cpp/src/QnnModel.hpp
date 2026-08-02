#ifndef QNNMODEL_HPP
#define QNNMODEL_HPP

#include <HTP/QnnHtpContext.h>
#include <HTP/QnnHtpDevice.h>
#include <dlfcn.h>
#include <inttypes.h>

#include <Config.hpp>
#include <QnnSampleApp.hpp>
#include <QnnTypeMacros.hpp>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

#include "DataUtil.hpp"
#include "Logger.hpp"
#include "SDUtils.hpp"

using namespace qnn::tools::sample_app;

class QnnModel : public QnnSampleApp {
 public:
  Qnn_Tensor_t *inputs = nullptr;
  Qnn_Tensor_t *outputs = nullptr;
  void *m_modelHandle = nullptr;
  bool io_logged_ = false;
  QnnModel(QnnFunctionPointers qnnFunctionPointers, std::string inputListPaths,
           std::string opPackagePaths, void *backendHandle,
           std::string outputPath = s_defaultOutputPath, bool debug = false,
           qnn::tools::iotensor::OutputDataType outputDataType =
               qnn::tools::iotensor::OutputDataType::FLOAT_ONLY,
           qnn::tools::iotensor::InputDataType inputDataType =
               qnn::tools::iotensor::InputDataType::FLOAT,
           ProfilingLevel profilingLevel = ProfilingLevel::OFF,
           bool dumpOutputs = false, std::string cachedBinaryPath = "",
           std::string saveBinaryName = "")
      : QnnSampleApp(qnnFunctionPointers, inputListPaths, opPackagePaths,
                     backendHandle, outputPath, debug, outputDataType,
                     inputDataType, profilingLevel, dumpOutputs,
                     cachedBinaryPath, saveBinaryName) {}

  // --- HTP spill-fill buffer sharing across contexts (group registration) ---
  // When enabled, this context joins a group that shares a single HTP
  // spill-fill scratch buffer instead of each context allocating its own.
  // The first model in the group passes firstHandle == nullptr (it becomes the
  // group head); every subsequent model passes the head's context handle
  // (getContextHandle()). `maxBytes` must be >= the largest spill-fill
  // requirement among all models in the group; only the head's value is
  // honored by the backend. Must be called BEFORE initialize()/createFromBinary
  // so the config is in place when the context is created.
  // NOTE: old (pre-2.35) context binaries report spillFillBufferSize == 0 in
  // their metadata, so the size cannot be read back from them — it must be
  // supplied here explicitly.
  void setSpillFillGroup(uint64_t maxBytes, Qnn_ContextHandle_t firstHandle) {
    if (maxBytes == 0) return;  // disabled: leave m_contextConfig untouched
    m_sfHtpConfig.option =
        QNN_HTP_CONTEXT_CONFIG_OPTION_REGISTER_MULTI_CONTEXTS;
    m_sfHtpConfig.groupRegistration.firstGroupHandle = firstHandle;
    m_sfHtpConfig.groupRegistration.maxSpillFillBuffer = maxBytes;
    m_sfCtxConfig.option = QNN_CONTEXT_CONFIG_OPTION_CUSTOM;
    m_sfCtxConfig.customConfig = &m_sfHtpConfig;
    m_sfCtxConfigPtrs[0] = &m_sfCtxConfig;
    m_sfCtxConfigPtrs[1] = nullptr;
    // Consumed by QnnSampleApp::createFromBinary / QnnModel::createFromBuffer.
    m_contextConfig = m_sfCtxConfigPtrs;
  }

  // Valid only after a successful initialize()/createFromBinary(): the QNN
  // context handle, used as the group head reference for the other models.
  Qnn_ContextHandle_t getContextHandle() const { return m_context; }

  // Queries the HTP backend for this context's real spill-fill scratch
  // requirement (bytes). Valid only after the context is created. Returns 0 if
  // the query is unavailable/fails. Use this to discover the right value for
  // setSpillFillGroup() instead of guessing — note the spillFillBufferSize in
  // the binary metadata is 0 for pre-2.35 binaries, but this runtime query
  // returns the true requirement.
  uint64_t querySpillFillSize() const {
    if (m_context == nullptr ||
        m_qnnFunctionPointers.qnnInterface.contextGetProperty == nullptr)
      return 0;
    QnnHtpContext_CustomProperty_t custom{};
    custom.option = QNN_HTP_CONTEXT_GET_PROP_MAX_SPILLFILL_BUFFER_SIZE;
    QnnContext_Property_t prop{};
    prop.option = QNN_CONTEXT_PROPERTY_OPTION_CUSTOM;
    prop.customProperty = &custom;
    QnnContext_Property_t *props[2] = {&prop, nullptr};
    if (QNN_SUCCESS !=
        m_qnnFunctionPointers.qnnInterface.contextGetProperty(m_context, props))
      return 0;
    return custom.spillfillBufferSize;
  }

  ~QnnModel() {
    // Tear down per-graph input/output tensors first (allocated lazily by
    // setupInputAndOutputTensors). Must run before freeContext() since it
    // relies on graphsInfo for tensor counts.
    if ((inputs != nullptr || outputs != nullptr) && m_graphsInfo != nullptr &&
        m_graphsCount > 0) {
      m_ioTensor.tearDownInputAndOutputTensors(
          inputs, outputs, (*m_graphsInfo)[0].numInputTensors,
          (*m_graphsInfo)[0].numOutputTensors);
    }
    inputs = nullptr;
    outputs = nullptr;

    // freeContext() is not idempotent — only call when graphs are still alive.
    if (m_graphsInfo != nullptr) {
      freeContext();
    }
    freeDevice();
    terminateBackend();
    if (m_modelHandle != nullptr) {
      dlclose(m_modelHandle);
      m_modelHandle = nullptr;
    }
  }

  StatusCode enablePerformaceMode() {
    uint32_t powerConfigId;
    uint32_t deviceId = 0;
    uint32_t coreId = 0;
    auto qnnInterface = m_qnnFunctionPointers.qnnInterface;

    QnnDevice_Infrastructure_t deviceInfra = nullptr;
    Qnn_ErrorHandle_t devErr =
        qnnInterface.deviceGetInfrastructure(&deviceInfra);
    if (devErr != QNN_SUCCESS) {
      QNN_ERROR("device error");
      return StatusCode::FAILURE;
    }
    QnnHtpDevice_Infrastructure_t *htpInfra =
        static_cast<QnnHtpDevice_Infrastructure_t *>(deviceInfra);
    QnnHtpDevice_PerfInfrastructure_t perfInfra = htpInfra->perfInfra;
    Qnn_ErrorHandle_t perfInfraErr =
        perfInfra.createPowerConfigId(deviceId, coreId, &powerConfigId);
    if (perfInfraErr != QNN_SUCCESS) {
      QNN_ERROR("createPowerConfigId failed");
      return StatusCode::FAILURE;
    }
    QnnHtpPerfInfrastructure_PowerConfig_t rpcControlLatency;
    memset(&rpcControlLatency, 0, sizeof(rpcControlLatency));
    rpcControlLatency.option =
        QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_CONTROL_LATENCY;
    rpcControlLatency.rpcControlLatencyConfig = 100;
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs1[] = {
        &rpcControlLatency, NULL};
    perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs1);
    if (perfInfraErr != QNN_SUCCESS) {
      QNN_ERROR("setPowerConfig failed");
      return StatusCode::FAILURE;
    }

    QnnHtpPerfInfrastructure_PowerConfig_t rpcPollingTime;
    memset(&rpcPollingTime, 0, sizeof(rpcPollingTime));
    rpcPollingTime.option =
        QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_RPC_POLLING_TIME;
    rpcPollingTime.rpcPollingTimeConfig = 9999;
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs2[] = {
        &rpcPollingTime, NULL};
    perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs2);
    if (perfInfraErr != QNN_SUCCESS) {
      QNN_ERROR("setPowerConfig failed");
      return StatusCode::FAILURE;
    }

    QnnHtpPerfInfrastructure_PowerConfig_t powerConfig;
    memset(&powerConfig, 0, sizeof(powerConfig));
    powerConfig.option = QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_DCVS_V3;
    powerConfig.dcvsV3Config.dcvsEnable = 0;
    powerConfig.dcvsV3Config.setDcvsEnable = 1;
    powerConfig.dcvsV3Config.contextId = powerConfigId;
    powerConfig.dcvsV3Config.powerMode =
        QNN_HTP_PERF_INFRASTRUCTURE_POWERMODE_PERFORMANCE_MODE;
    powerConfig.dcvsV3Config.setSleepLatency = 1;
    powerConfig.dcvsV3Config.setBusParams = 1;
    powerConfig.dcvsV3Config.setCoreParams = 1;
    powerConfig.dcvsV3Config.sleepDisable = 1;
    powerConfig.dcvsV3Config.setSleepDisable = 1;
    powerConfig.dcvsV3Config.sleepLatency = 40;
    powerConfig.dcvsV3Config.busVoltageCornerMin =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.busVoltageCornerTarget =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.busVoltageCornerMax =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerMin =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerTarget =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    powerConfig.dcvsV3Config.coreVoltageCornerMax =
        DCVS_VOLTAGE_VCORNER_MAX_VOLTAGE_CORNER;
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs3[] = {
        &powerConfig, NULL};
    perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs3);
    if (perfInfraErr != QNN_SUCCESS) {
      QNN_ERROR("setPowerConfig failed");
      return StatusCode::FAILURE;
    }

    QnnHtpPerfInfrastructure_PowerConfig_t adaptivePollingTime;
    memset(&adaptivePollingTime, 0, sizeof(adaptivePollingTime));
    adaptivePollingTime.option =
        QNN_HTP_PERF_INFRASTRUCTURE_POWER_CONFIGOPTION_ADAPTIVE_POLLING_TIME;
    adaptivePollingTime.adaptivePollingTimeConfig = 1000;
    const QnnHtpPerfInfrastructure_PowerConfig_t *powerConfigs4[] = {
        &adaptivePollingTime, NULL};
    perfInfraErr = perfInfra.setPowerConfig(powerConfigId, powerConfigs4);
    if (perfInfraErr != QNN_SUCCESS) {
      QNN_ERROR("setPowerConfig failed");
      return StatusCode::FAILURE;
    }

    return StatusCode::SUCCESS;
  }

  StatusCode executeUnetGraphs(float *latents, int timestep,
                               float *text_embedding, float *latents_pred) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting unet execution for graphIdx: %d", graphIdx);

    // set input/output tensor
    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 3) {
      QNN_ERROR("Expecting 3 input tensors, got %d", graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // latents
    {
      uint16_t *latents_uint16 =
          static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data);
      int elementCount = 1 * 4 * sample_width * sample_height;
      qnn::tools::datautil::floatToTfN(
          latents_uint16, latents,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale, elementCount);
    }

    // position/timestep
    {
      int32_t *positionData =
          static_cast<int32_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[1]).data);
      positionData[0] = timestep;
    }

    // text_embedding
    {
      uint16_t *text_embedding_uint16 =
          static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[2]).data);
      int elementCount = 1 * 77 * text_embedding_size;
      qnn::tools::datautil::floatToTfN(
          text_embedding_uint16, text_embedding,
          inputs[2].v1.quantizeParams.scaleOffsetEncoding.offset,
          inputs[2].v1.quantizeParams.scaleOffsetEncoding.scale, elementCount);
    }

    // execute graph
    QNN_DEBUG("Executing unet graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("unet graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("unet graph execution failed!");
    }

    // get output
    if (StatusCode::SUCCESS == returnStatus) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.convertToFloatInto(latents_pred, &outputs[0])) {
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }

    return returnStatus;
  }

  StatusCode executeVaeEncoderGraphs(float *pixel_values, float *mean,
                                     float *std) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting vae encoder execution for graphIdx: %d", graphIdx);

    // set input/output tensor
    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 1) {
      QNN_ERROR("Expecting 1 input tensors, got %d", graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // pixel_values
    {
      uint16_t *pixel_values_uint16 =
          static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data);
      int elementCount = 1 * 3 * output_width * output_height;
      qnn::tools::datautil::floatToTfN(
          pixel_values_uint16, pixel_values,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale, elementCount);
    }

    // execute graph
    QNN_DEBUG("Executing vae encoder graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("vae encoder graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("vae encoder graph execution failed!");
    }

    // get output
    if (StatusCode::SUCCESS == returnStatus) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.convertToFloatInto(mean, &outputs[0])) {
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.convertToFloatInto(std, &outputs[1])) {
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    return returnStatus;
  }

  StatusCode executeVaeDecoderGraphs(float *latents, float *pixel_values) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting vae decoder execution for graphIdx: %d", graphIdx);

    // set input/output tensor
    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 1) {
      QNN_ERROR("Expecting 1 input tensors, got %d", graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // latents
    {
      uint16_t *latents_uint16 =
          static_cast<uint16_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data);
      int elementCount = 1 * 4 * sample_width * sample_height;
      qnn::tools::datautil::floatToTfN(
          latents_uint16, latents,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset,
          inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale, elementCount);
    }

    // execute graph
    QNN_DEBUG("Executing vae decoder graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("vae decoder graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("vae decoder graph execution failed!");
    }

    // get output
    if (StatusCode::SUCCESS == returnStatus) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.convertToFloatInto(pixel_values, &outputs[0])) {
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    return returnStatus;
  }

  StatusCode executeUnetGraphsSDXL(float *sample, int timestep,
                                   float *encoder_hidden_states,
                                   float *text_embeds, float *time_ids,
                                   float *out_sample) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting sdxl unet execution for graphIdx: %d", graphIdx);

    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 5) {
      QNN_ERROR("Expecting 5 input tensors for sdxl unet, got %d",
                graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // sample (fp32, 1x4xHxW)
    {
      int elementCount = 1 * 4 * sample_width * sample_height;
      memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data),
             sample, elementCount * sizeof(float));
    }

    // timestep (int32, 1)
    {
      int32_t *ts =
          static_cast<int32_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[2]).data);
      ts[0] = timestep;
    }

    // encoder_hidden_states (fp32, 1x77x2048)
    {
      int elementCount = 1 * 77 * (text_embedding_size + text_embedding_size_2);
      memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[1]).data),
             encoder_hidden_states, elementCount * sizeof(float));
    }

    // text_embeds (fp32, 1x1280)
    {
      int elementCount = 1 * text_embedding_size_2;
      memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[4]).data),
             text_embeds, elementCount * sizeof(float));
    }

    // time_ids (fp32, 1x6)
    {
      int elementCount = 1 * 6;
      memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[3]).data),
             time_ids, elementCount * sizeof(float));
    }

    QNN_DEBUG("Executing sdxl unet graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("sdxl unet graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("sdxl unet graph execution failed!");
      return returnStatus;
    }

    // out_sample (fp32, 1x4xHxW)
    int elementCount = 1 * 4 * sample_width * sample_height;
    memcpy(out_sample,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           elementCount * sizeof(float));

    return returnStatus;
  }

  StatusCode executeVaeEncoderGraphsSDXL(float *pixel_values, float *mean,
                                         float *std) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting sdxl vae encoder execution for graphIdx: %d", graphIdx);

    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 1) {
      QNN_ERROR("Expecting 1 input tensor for sdxl vae encoder, got %d",
                graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // pixel_values (fp32, 1x3xHxW)
    {
      int elementCount = 1 * 3 * output_width * output_height;
      memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data),
             pixel_values, elementCount * sizeof(float));
    }

    QNN_DEBUG("Executing sdxl vae encoder graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("sdxl vae encoder graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("sdxl vae encoder graph execution failed!");
      return returnStatus;
    }

    int elementCount = 1 * 4 * sample_width * sample_height;
    memcpy(mean,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           elementCount * sizeof(float));
    memcpy(std,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[1]).data),
           elementCount * sizeof(float));

    return returnStatus;
  }

  StatusCode executeVaeDecoderGraphsSDXL(float *latents, float *pixel_values) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting sdxl vae decoder execution for graphIdx: %d", graphIdx);

    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 1) {
      QNN_ERROR("Expecting 1 input tensor for sdxl vae decoder, got %d",
                graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // latents (fp32, 1x4xHxW)
    {
      int elementCount = 1 * 4 * sample_width * sample_height;
      memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data),
             latents, elementCount * sizeof(float));
    }

    QNN_DEBUG("Executing sdxl vae decoder graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("sdxl vae decoder graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("sdxl vae decoder graph execution failed!");
      return returnStatus;
    }

    int elementCount = 1 * 3 * output_width * output_height;
    memcpy(pixel_values,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           elementCount * sizeof(float));

    return returnStatus;
  }

  // ---- shared helpers for the Anima execute paths --------------------------
  bool ensureIoTensors() {
    if (inputs != nullptr && outputs != nullptr) return true;
    if (qnn::tools::iotensor::StatusCode::SUCCESS !=
        m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                              (*m_graphsInfo)[0])) {
      QNN_ERROR("Error setting up Input/Output tensors");
      return false;
    }
    return true;
  }

  static size_t tensorElems(const Qnn_Tensor_t &t) {
    uint32_t rank = QNN_TENSOR_GET_RANK(t);
    uint32_t *dims = QNN_TENSOR_GET_DIMENSIONS(t);
    size_t n = 1;
    for (uint32_t i = 0; i < rank; ++i) n *= (dims ? dims[i] : 1);
    return n;
  }

  static Qnn_Tensor_t *findTensor(Qnn_Tensor_t *arr, uint32_t count,
                                  const char *name) {
    for (uint32_t i = 0; i < count; ++i) {
      const char *tn = QNN_TENSOR_GET_NAME(arr[i]);
      if (tn && std::strcmp(tn, name) == 0) return &arr[i];
    }
    return nullptr;
  }

  bool writeNamedFloat(const qnn_wrapper_api::GraphInfo_t &graphInfo,
                       const char *name, const float *src, size_t elems) {
    Qnn_Tensor_t *t = findTensor(inputs, graphInfo.numInputTensors, name);
    if (!t) {
      QNN_ERROR("missing input tensor '%s'", name);
      return false;
    }
    // A shape the caller and the graph disagree on would otherwise be a silent
    // heap overrun. It is a real risk for the split-DiT paths, where the
    // handoff sizes come from whatever the conversion produced rather than from
    // constants in this file.
    const size_t capacity = tensorElems(*t);
    if (elems != capacity) {
      QNN_ERROR("input tensor '%s' expects %zu elements, got %zu", name,
                capacity, elems);
      return false;
    }
    memcpy(QNN_TENSOR_GET_CLIENT_BUF(*t).data, src, elems * sizeof(float));
    return true;
  }

  bool runGraph(const qnn_wrapper_api::GraphInfo_t &graphInfo,
                const char *tag) {
    auto start = std::chrono::high_resolution_clock::now();
    auto st = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);
    auto end = std::chrono::high_resolution_clock::now();
    QNN_INFO(
        "%s graph execution time: %d ms", tag,
        (int)std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
            .count());
    if (QNN_GRAPH_NO_ERROR != st) {
      QNN_ERROR("%s graph execution failed!", tag);
      return false;
    }
    return true;
  }

  // One-time dump of a graph's IO names+sizes, so the on-device tensor naming
  // can be verified against export_onnx_anima.py's input/output_names.
  void logGraphIoOnce(const char *format, const char *tag,
                      const qnn_wrapper_api::GraphInfo_t &graphInfo) {
    if (io_logged_) return;
    io_logged_ = true;
    for (uint32_t i = 0; i < graphInfo.numInputTensors; ++i)
      QNN_INFO("[%s %s] in[%u] name=%s elems=%zu", format, tag, i,
               QNN_TENSOR_GET_NAME(inputs[i]), tensorElems(inputs[i]));
    for (uint32_t i = 0; i < graphInfo.numOutputTensors; ++i)
      QNN_INFO("[%s %s] out[%u] name=%s elems=%zu", format, tag, i,
               QNN_TENSOR_GET_NAME(outputs[i]), tensorElems(outputs[i]));
  }

  void logAnimaIoOnce(const char *tag,
                      const qnn_wrapper_api::GraphInfo_t &graphInfo) {
    logGraphIoOnce("anima", tag, graphInfo);
  }

  // ---- Anima (split DiT + 16-ch Wan VAE) -----------------------------------
  // Anima graphs are converted with --preserve_io_datatype, so every IO tensor
  // is fp32 (direct memcpy, like the SDXL path). The split DiT crosses only TWO
  // named intermediates between part-1 and part-2 (hidden, emb); part-2
  // recomputes adaln + rope internally from `timestamp` (and the static grid),
  // and the app re-supplies `context`. Passing adaln/rope as flat graph inputs
  // used to force part-2's residual stream into a slow layout (~+200M
  // cycles/block); computing them inside removes it. We bind every tensor by
  // its (ONNX) name. Matches export_onnx_anima_scaled.py PART1_OUT_NAMES.
  static constexpr const char *kAnimaPart1OutNames[2] = {"hidden", "emb"};

  // part-1: (sample, timestamp, encoder_hidden_states) -> (hidden, emb).
  StatusCode executeAnimaUnetPart1(float *sample, float timestep,
                                   float *context,
                                   std::vector<std::vector<float>> &out2) {
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    auto graphInfo = (*m_graphsInfo)[0];
    logAnimaIoOnce("unet_part1", graphInfo);

    const int latent_elems =
        1 * anima_latent_channels * sample_width * sample_height;
    const int ctx_elems = 1 * anima_text_seq_len * anima_text_embedding_size;
    if (!writeNamedFloat(graphInfo, "sample", sample, latent_elems) ||
        !writeNamedFloat(graphInfo, "timestamp", &timestep, 1) ||
        !writeNamedFloat(graphInfo, "encoder_hidden_states", context,
                         ctx_elems))
      return StatusCode::FAILURE;

    if (!runGraph(graphInfo, "anima unet part1")) return StatusCode::FAILURE;

    out2.resize(2);
    for (int k = 0; k < 2; ++k) {
      Qnn_Tensor_t *t = findTensor(outputs, graphInfo.numOutputTensors,
                                   kAnimaPart1OutNames[k]);
      if (!t) {
        QNN_ERROR("anima part1: missing output '%s'", kAnimaPart1OutNames[k]);
        return StatusCode::FAILURE;
      }
      size_t n = tensorElems(*t);
      out2[k].resize(n);
      memcpy(out2[k].data(), QNN_TENSOR_GET_CLIENT_BUF(*t).data,
             n * sizeof(float));
    }
    return StatusCode::SUCCESS;
  }

  // part-2: (hidden, emb) + timestamp + context -> out_sample (1x16xHxW).
  // adaln + rope are recomputed inside the graph from `timestamp`/grid.
  StatusCode executeAnimaUnetPart2(const std::vector<std::vector<float>> &in2,
                                   float timestep, float *context,
                                   float *out_sample) {
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    auto graphInfo = (*m_graphsInfo)[0];
    logAnimaIoOnce("unet_part2", graphInfo);

    const int ctx_elems = 1 * anima_text_seq_len * anima_text_embedding_size;
    if (!writeNamedFloat(graphInfo, "hidden", in2[0].data(),
                         (int)in2[0].size()) ||
        !writeNamedFloat(graphInfo, "emb", in2[1].data(), (int)in2[1].size()) ||
        !writeNamedFloat(graphInfo, "timestamp", &timestep, 1) ||
        !writeNamedFloat(graphInfo, "context", context, ctx_elems))
      return StatusCode::FAILURE;

    if (!runGraph(graphInfo, "anima unet part2")) return StatusCode::FAILURE;

    const int latent_elems =
        1 * anima_latent_channels * sample_width * sample_height;
    memcpy(out_sample,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           latent_elems * sizeof(float));
    return StatusCode::SUCCESS;
  }

  // 16-ch latent -> 3-ch pixels. Same shape contract as the SDXL decoder but
  // with 16 input channels.
  StatusCode executeAnimaVaeDecoder(float *latents, float *pixel_values) {
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    auto graphInfo = (*m_graphsInfo)[0];
    logAnimaIoOnce("vae_decoder", graphInfo);

    const int latent_elems =
        1 * anima_latent_channels * sample_width * sample_height;
    memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data),
           latents, latent_elems * sizeof(float));

    if (!runGraph(graphInfo, "anima vae decoder")) return StatusCode::FAILURE;

    const int pixel_elems = 1 * 3 * output_width * output_height;
    memcpy(pixel_values,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           pixel_elems * sizeof(float));
    return StatusCode::SUCCESS;
  }

  // 3-ch pixels (1x3xHxW, -1..1) -> 16-ch latent distribution (mean, std).
  // Same single-input / two-output contract as the SDXL VAE encoder, but with
  // 16 latent channels. Used for Anima img2img / inpaint / aspect-pad.
  StatusCode executeAnimaVaeEncoder(float *pixel_values, float *mean,
                                    float *std_dev) {
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    auto graphInfo = (*m_graphsInfo)[0];
    logAnimaIoOnce("vae_encoder", graphInfo);

    const int pixel_elems = 1 * 3 * output_width * output_height;
    memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data),
           pixel_values, pixel_elems * sizeof(float));

    if (!runGraph(graphInfo, "anima vae encoder")) return StatusCode::FAILURE;

    const int latent_elems =
        1 * anima_latent_channels * sample_width * sample_height;
    memcpy(mean,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           latent_elems * sizeof(float));
    memcpy(std_dev,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[1]).data),
           latent_elems * sizeof(float));
    return StatusCode::SUCCESS;
  }

  // Merged Qwen3 + LLM-adapter text encoder. Inputs: input_embedding
  // [1,QWEN_SEQ,1024] fp32, t5_ids [1,T5_SEQ] int32, t5_mask [1,T5_SEQ] fp32,
  // qwen_mask [1,QWEN_SEQ] fp32. Output: context [1,T5_SEQ,1024] fp32. (t5_ids
  // is declared int32 in the graph thanks to --preserve_io_datatype, and the
  // C++ QNN API takes a real int32 buffer — unlike qnn-net-run which wants
  // float32 raw.)
  StatusCode executeAnimaClip(const float *input_embedding,
                              const int32_t *t5_ids, const float *t5_mask,
                              const float *qwen_mask, float *out_context) {
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    auto graphInfo = (*m_graphsInfo)[0];
    logAnimaIoOnce("clip", graphInfo);

    const int QS = anima_qwen_seq_len;
    const int TS = anima_text_seq_len;
    const int D = anima_text_embedding_size;
    if (!writeNamedFloat(graphInfo, "input_embedding", input_embedding,
                         (size_t)QS * D) ||
        !writeNamedFloat(graphInfo, "t5_mask", t5_mask, TS) ||
        !writeNamedFloat(graphInfo, "qwen_mask", qwen_mask, QS))
      return StatusCode::FAILURE;
    // t5_ids is int32 (not float) — bind it directly.
    Qnn_Tensor_t *ti = findTensor(inputs, graphInfo.numInputTensors, "t5_ids");
    if (!ti) {
      QNN_ERROR("anima clip: missing input 't5_ids'");
      return StatusCode::FAILURE;
    }
    memcpy(QNN_TENSOR_GET_CLIENT_BUF(*ti).data, t5_ids,
           (size_t)TS * sizeof(int32_t));

    if (!runGraph(graphInfo, "anima clip")) return StatusCode::FAILURE;

    Qnn_Tensor_t *to =
        findTensor(outputs, graphInfo.numOutputTensors, "context");
    if (!to) {
      QNN_ERROR("anima clip: missing output 'context'");
      return StatusCode::FAILURE;
    }
    memcpy(out_context, QNN_TENSOR_GET_CLIENT_BUF(*to).data,
           (size_t)TS * D * sizeof(float));
    return StatusCode::SUCCESS;
  }

  // ---- Z-Image (S3-DiT, N-way split + Flux 16-ch VAE) ----------------------
  // The S3-DiT is single-stream: text and image tokens live in ONE residual
  // sequence, so unlike Anima's split there is no separate `context` to
  // re-supply past the first part — the caption tokens are already inside
  // `hidden`. Two things do have to reach every part:
  //
  // The unified sequence is ordered [image tokens, caption slots] — image
  // FIRST, which is what diffusers' basic mode does.
  //
  //   pos_ids   3D RoPE coordinates (t, h, w) per token. These CANNOT be baked
  //             into the graph: the reference drops padded caption rows, so the
  //             caption length varies per prompt and image tokens sit at
  //             cap_len + 1, moving every image token's t with the prompt.
  //   attn_mask 1 for tokens the reference would have created, 0 beyond. The
  //             graph MUST apply this in two places — the caption refiner and
  //             the main blocks. With both, a fixed 512-slot caption is
  //             bit-exact against the reference; with only the main blocks it
  //             is not (docs/zimage.md section 9).
  //   cap_pad_mask (part 1 only) 1 where a caption row must be replaced by the
  //             DiT's learned pad token, i.e. past the real prompt.
  //
  // `emb` (the timestep adaLN vector) is computed once by part 1 and reused by
  // every block, so later parts take it directly and never see `timestep`.
  //
  // 6B parameters do not fit a single HTP context at any supported weight
  // width, so the DiT is exported as N pieces cut between transformer blocks.
  // The chain is uniform and N is discovered on disk, not fixed here:
  //   part 1    : (sample, timestep, context, pos_ids, attn_mask,
  //                cap_pad_mask)                        -> (hidden, emb)
  //   part 1<k<N: (hidden, emb, pos_ids, attn_mask)      -> (hidden)
  //   part N    : (hidden, emb, pos_ids, attn_mask)      -> (out_sample)
  // A part is recognised as terminal purely by exposing an output named
  // `out_sample`, so a single-context export (N = 1, i.e. part 1 terminal)
  // needs no special case.
  // The residual stream is named differently on the way in and on the way out,
  // and that is forced rather than stylistic: torch.onnx.export cannot give a
  // graph an input and an output with the same name, and silently renames the
  // input to "hidden.1" if you try. Binding is by name, so that rename would
  // only surface on device as a missing tensor. Inputs and outputs are separate
  // arrays in GraphInfo_t, so nothing here depends on the two matching.
  static constexpr const char *kZImageStateInNames[2] = {"hidden_in", "emb"};
  static constexpr const char *kZImageStateOutNames[2] = {"hidden", "emb"};
  static constexpr const char *kZImageOutName = "out_sample";

  // Collects whatever the part produced. A terminal part writes the velocity
  // into out_sample; any other part refreshes `hidden` (and `emb`, if that part
  // chose to re-emit it rather than let the host re-supply the unchanged one).
  StatusCode readZImageDitOutputs(const qnn_wrapper_api::GraphInfo_t &graphInfo,
                                  std::vector<std::vector<float>> &state,
                                  float *out_sample) {
    Qnn_Tensor_t *term =
        findTensor(outputs, graphInfo.numOutputTensors, kZImageOutName);
    if (term) {
      if (!out_sample) {
        QNN_ERROR("zimage dit: terminal part reached with no output buffer");
        return StatusCode::FAILURE;
      }
      const size_t latent_elems =
          (size_t)zimage_latent_channels * sample_width * sample_height;
      memcpy(out_sample, QNN_TENSOR_GET_CLIENT_BUF(*term).data,
             latent_elems * sizeof(float));
      return StatusCode::SUCCESS;
    }

    state.resize(2);
    bool saw_hidden = false;
    for (int k = 0; k < 2; ++k) {
      Qnn_Tensor_t *t = findTensor(outputs, graphInfo.numOutputTensors,
                                   kZImageStateOutNames[k]);
      // `emb` is constant for the whole chain, so only part 1 has to emit it;
      // later parts may omit it and keep the copy the host already holds.
      if (!t) continue;
      size_t n = tensorElems(*t);
      state[k].resize(n);
      memcpy(state[k].data(), QNN_TENSOR_GET_CLIENT_BUF(*t).data,
             n * sizeof(float));
      if (k == 0) saw_hidden = true;
    }
    if (!saw_hidden) {
      QNN_ERROR("zimage dit: part emitted neither '%s' nor '%s'",
                kZImageOutName, kZImageStateOutNames[0]);
      return StatusCode::FAILURE;
    }
    if (state[1].empty()) {
      QNN_ERROR("zimage dit: '%s' never produced by any part",
                kZImageStateOutNames[1]);
      return StatusCode::FAILURE;
    }
    return StatusCode::SUCCESS;
  }

  // True when this graph ends the chain (emits the velocity rather than a
  // residual-stream handoff).
  bool zimageGraphIsTerminal() {
    if (!ensureIoTensors()) return false;
    auto graphInfo = (*m_graphsInfo)[0];
    return findTensor(outputs, graphInfo.numOutputTensors, kZImageOutName) !=
           nullptr;
  }

  // Binds the per-token RoPE coordinates (int32, bound directly like Anima's
  // t5_ids) and the attention mask. Shared by every part of the chain.
  bool writeZImagePositions(const qnn_wrapper_api::GraphInfo_t &graphInfo,
                            const int32_t *pos_ids, const float *attn_mask,
                            size_t tokens) {
    Qnn_Tensor_t *pi =
        findTensor(inputs, graphInfo.numInputTensors, "pos_ids");
    if (!pi) {
      QNN_ERROR("zimage dit: missing input 'pos_ids'");
      return false;
    }
    const size_t want = tokens * zimage_rope_axes;
    const size_t capacity = tensorElems(*pi);
    if (want != capacity) {
      QNN_ERROR("zimage dit: 'pos_ids' expects %zu elements, got %zu", capacity,
                want);
      return false;
    }
    memcpy(QNN_TENSOR_GET_CLIENT_BUF(*pi).data, pos_ids,
           want * sizeof(int32_t));
    return writeNamedFloat(graphInfo, "attn_mask", attn_mask, tokens);
  }

  // part 1: (sample, timestep, context, pos_ids, attn_mask) -> (hidden, emb),
  // or straight to out_sample when the whole DiT fits one context.
  StatusCode executeZImageDitFirst(const float *sample, float timestep,
                                   const float *context,
                                   const int32_t *pos_ids,
                                   const float *attn_mask,
                                   const float *cap_pad_mask, size_t tokens,
                                   std::vector<std::vector<float>> &state,
                                   float *out_sample) {
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    auto graphInfo = (*m_graphsInfo)[0];
    logGraphIoOnce("zimage", "dit_part1", graphInfo);

    const size_t latent_elems =
        (size_t)zimage_latent_channels * sample_width * sample_height;
    const size_t ctx_elems =
        (size_t)zimage_text_seq_len * zimage_text_embedding_size;
    if (!writeNamedFloat(graphInfo, "sample", sample, latent_elems) ||
        !writeNamedFloat(graphInfo, "timestep", &timestep, 1) ||
        !writeNamedFloat(graphInfo, "context", context, ctx_elems) ||
        !writeNamedFloat(graphInfo, "cap_pad_mask", cap_pad_mask,
                         zimage_text_seq_len) ||
        !writeZImagePositions(graphInfo, pos_ids, attn_mask, tokens))
      return StatusCode::FAILURE;

    if (!runGraph(graphInfo, "zimage dit part1")) return StatusCode::FAILURE;
    return readZImageDitOutputs(graphInfo, state, out_sample);
  }

  // Every part after the first. `state` carries {hidden, emb} in and is updated
  // in place; a terminal part writes out_sample and leaves state untouched.
  // No `timestep` here: `emb` already IS the timestep's adaLN vector.
  StatusCode executeZImageDitNext(std::vector<std::vector<float>> &state,
                                  const int32_t *pos_ids,
                                  const float *attn_mask, size_t tokens,
                                  float *out_sample) {
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    auto graphInfo = (*m_graphsInfo)[0];
    logGraphIoOnce("zimage", "dit_partN", graphInfo);

    if (state.size() < 2 || state[0].empty() || state[1].empty()) {
      QNN_ERROR("zimage dit: residual state not populated by the prior part");
      return StatusCode::FAILURE;
    }
    if (!writeNamedFloat(graphInfo, kZImageStateInNames[0], state[0].data(),
                         state[0].size()) ||
        !writeNamedFloat(graphInfo, kZImageStateInNames[1], state[1].data(),
                         state[1].size()) ||
        !writeZImagePositions(graphInfo, pos_ids, attn_mask, tokens))
      return StatusCode::FAILURE;

    if (!runGraph(graphInfo, "zimage dit part")) return StatusCode::FAILURE;
    return readZImageDitOutputs(graphInfo, state, out_sample);
  }

  // Z-Image VAE decoder: 16-ch latent -> 3-ch pixels. Identical contract to
  // Anima's (both are 16-channel), kept as its own entry point so the two
  // formats' constants never cross.
  StatusCode executeZImageVaeDecoder(const float *latents,
                                     float *pixel_values) {
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    auto graphInfo = (*m_graphsInfo)[0];
    logGraphIoOnce("zimage", "vae_decoder", graphInfo);

    const size_t latent_elems =
        (size_t)zimage_latent_channels * sample_width * sample_height;
    memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data),
           latents, latent_elems * sizeof(float));

    if (!runGraph(graphInfo, "zimage vae decoder")) return StatusCode::FAILURE;

    const size_t pixel_elems = (size_t)3 * output_width * output_height;
    memcpy(pixel_values,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           pixel_elems * sizeof(float));
    return StatusCode::SUCCESS;
  }

  // 3-ch pixels (-1..1) -> 16-ch latent distribution (mean, std).
  StatusCode executeZImageVaeEncoder(const float *pixel_values, float *mean,
                                     float *std_dev) {
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    auto graphInfo = (*m_graphsInfo)[0];
    logGraphIoOnce("zimage", "vae_encoder", graphInfo);

    const size_t pixel_elems = (size_t)3 * output_width * output_height;
    memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data),
           pixel_values, pixel_elems * sizeof(float));

    if (!runGraph(graphInfo, "zimage vae encoder")) return StatusCode::FAILURE;

    const size_t latent_elems =
        (size_t)zimage_latent_channels * sample_width * sample_height;
    memcpy(mean,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           latent_elems * sizeof(float));
    memcpy(std_dev,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[1]).data),
           latent_elems * sizeof(float));
    return StatusCode::SUCCESS;
  }

  // Qwen3-4B text encoder. Inputs: input_embedding [1,512,2560] fp32 (the
  // host-side token_emb lookup, prompt weights already folded in) and
  // attention_mask [1,512] fp32. Output: context [1,512,2560] fp32, which the
  // exporter must take from hidden_states[-2] (Z-Image reads the
  // second-to-last layer, not the final one).
  StatusCode executeZImageTextEncoder(const float *input_embedding,
                                      const float *attention_mask,
                                      float *out_context) {
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    auto graphInfo = (*m_graphsInfo)[0];
    logGraphIoOnce("zimage", "text_encoder", graphInfo);

    const size_t S = zimage_text_seq_len;
    const size_t D = zimage_text_embedding_size;
    if (!writeNamedFloat(graphInfo, "input_embedding", input_embedding,
                         S * D) ||
        !writeNamedFloat(graphInfo, "attention_mask", attention_mask, S))
      return StatusCode::FAILURE;

    if (!runGraph(graphInfo, "zimage text encoder")) return StatusCode::FAILURE;
    return readZImageClipOutput(graphInfo, out_context, S * D)
               ? StatusCode::SUCCESS
               : StatusCode::FAILURE;
  }

  // Text encoder parts after the first. Qwen3-4B is split across contexts
  // because 3.5 B parameters cannot be quantized in one piece; the chain runs
  // once per prompt, so unlike the DiT the extra hand-offs are amortised over a
  // whole image rather than paid per step.
  //
  //   part 1    : (input_embedding, attention_mask) -> hidden
  //   part 1<k<N: (hidden_in, attention_mask)       -> hidden
  //   part N    : (hidden_in, attention_mask)       -> context
  //
  // The mask reaches every part because each one rebuilds the causal + padding
  // mask internally from it. "hidden_in" rather than "hidden" for the input,
  // for the same reason the DiT parts use it: ONNX cannot name an input and an
  // output the same thing.
  StatusCode executeZImageClipNext(const float *hidden_in,
                                   const float *attention_mask,
                                   float *out_hidden) {
    if (!ensureIoTensors()) return StatusCode::FAILURE;
    auto graphInfo = (*m_graphsInfo)[0];
    logGraphIoOnce("zimage", "text_encoder_partN", graphInfo);

    const size_t S = zimage_text_seq_len;
    const size_t D = zimage_text_embedding_size;
    if (!writeNamedFloat(graphInfo, "hidden_in", hidden_in, S * D) ||
        !writeNamedFloat(graphInfo, "attention_mask", attention_mask, S))
      return StatusCode::FAILURE;

    if (!runGraph(graphInfo, "zimage text encoder part"))
      return StatusCode::FAILURE;
    return readZImageClipOutput(graphInfo, out_hidden, S * D)
               ? StatusCode::SUCCESS
               : StatusCode::FAILURE;
  }

  // True when this graph ends the encoder chain (emits the final `context`
  // rather than a mid-stack hand-off). Mirrors zimageGraphIsTerminal for the
  // DiT, and exists for the same reason: a chain that is short by a part still
  // runs, and without this check its last hidden state would silently be used
  // as the text embedding.
  bool zimageClipGraphIsTerminal() {
    if (!ensureIoTensors()) return false;
    auto graphInfo = (*m_graphsInfo)[0];
    return findTensor(outputs, graphInfo.numOutputTensors, "context") != nullptr;
  }

  // The terminal part of the encoder chain emits "context"; every earlier part
  // emits "hidden". A single-part encoder is terminal, so it emits "context"
  // too and the same read serves both layouts.
  bool readZImageClipOutput(const qnn_wrapper_api::GraphInfo_t &graphInfo,
                            float *dst, size_t elems) {
    Qnn_Tensor_t *to =
        findTensor(outputs, graphInfo.numOutputTensors, "context");
    if (!to) to = findTensor(outputs, graphInfo.numOutputTensors, "hidden");
    if (!to) {
      QNN_ERROR("zimage text encoder: part emits neither 'context' nor "
                "'hidden'");
      return false;
    }
    if (tensorElems(*to) != elems) {
      QNN_ERROR("zimage text encoder: output has %zu elements, expected %zu",
                tensorElems(*to), elems);
      return false;
    }
    memcpy(dst, QNN_TENSOR_GET_CLIENT_BUF(*to).data, elems * sizeof(float));
    return true;
  }

  StatusCode executeUpscalerGraphs(float *input_image, float *output_image) {
    auto returnStatus = StatusCode::SUCCESS;

    size_t graphIdx = 0;
    QNN_DEBUG("Starting upscaler execution for graphIdx: %d", graphIdx);

    // set input/output tensor
    if (inputs == nullptr || outputs == nullptr) {
      if (qnn::tools::iotensor::StatusCode::SUCCESS !=
          m_ioTensor.setupInputAndOutputTensors(&inputs, &outputs,
                                                (*m_graphsInfo)[graphIdx])) {
        QNN_ERROR(
            "Error in setting up Input and output Tensors for graphIdx: %d",
            graphIdx);
        returnStatus = StatusCode::FAILURE;
        return returnStatus;
      }
    }
    auto graphInfo = (*m_graphsInfo)[graphIdx];

    if (graphInfo.numInputTensors != 1) {
      QNN_ERROR("Expecting 1 input tensors, got %d", graphInfo.numInputTensors);
      returnStatus = StatusCode::FAILURE;
      return returnStatus;
    }

    // input_image (quantized to uint8, 1x3x192x192)
    {
      // uint8_t *input_uint8 =
      //     static_cast<uint8_t *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data);
      // int elementCount = 1 * 3 * 192 * 192;
      // qnn::tools::datautil::floatToTfN(
      //     input_uint8, input_image,
      //     inputs[0].v1.quantizeParams.scaleOffsetEncoding.offset,
      //     inputs[0].v1.quantizeParams.scaleOffsetEncoding.scale,
      //     elementCount);
      memcpy(static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(inputs[0]).data),
             input_image, 1 * 3 * 192 * 192 * sizeof(float));
    }

    // execute graph
    QNN_DEBUG("Executing upscaler graph: %d", graphIdx);
    auto start_time = std::chrono::high_resolution_clock::now();

    auto executeStatus = m_qnnFunctionPointers.qnnInterface.graphExecute(
        graphInfo.graph, inputs, graphInfo.numInputTensors, outputs,
        graphInfo.numOutputTensors, m_profileBackendHandle, nullptr);

    auto end_time = std::chrono::high_resolution_clock::now();
    int duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                       end_time - start_time)
                       .count();
    QNN_INFO("upscaler graph execution time: %d ms", duration);

    if (QNN_GRAPH_NO_ERROR != executeStatus) {
      returnStatus = StatusCode::FAILURE;
      QNN_ERROR("upscaler graph execution failed!");
    }

    // get output
    // if (StatusCode::SUCCESS == returnStatus) {
    //   float *tmp = nullptr;
    //   int elementCount = 1 * 3 * 768 * 768;
    //   if (qnn::tools::iotensor::StatusCode::SUCCESS !=
    //       m_ioTensor.convertToFloat(&tmp, &outputs[0])) {
    //     returnStatus = StatusCode::FAILURE;
    //     return returnStatus;
    //   }
    //   memcpy(output_image, tmp, elementCount * sizeof(float));
    //   free(tmp);
    // }
    memcpy(output_image,
           static_cast<float *>(QNN_TENSOR_GET_CLIENT_BUF(outputs[0]).data),
           1 * 3 * 768 * 768 * sizeof(float));
    return returnStatus;
  }

  StatusCode createFromBuffer(const uint8_t *buffer, uint64_t bufferSize) {
    if (nullptr == buffer || 0 == bufferSize) {
      QNN_ERROR("Invalid buffer provided. Buffer is null or size is 0.");
      return StatusCode::FAILURE;
    }

    if (nullptr ==
            m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate ||
        nullptr == m_qnnFunctionPointers.qnnSystemInterface
                       .systemContextGetBinaryInfo ||
        nullptr == m_qnnFunctionPointers.qnnSystemInterface.systemContextFree) {
      QNN_ERROR("QNN System function pointers are not populated.");
      return StatusCode::FAILURE;
    }

    auto returnStatus = StatusCode::SUCCESS;
    QnnSystemContext_Handle_t sysCtxHandle{nullptr};

    if (QNN_SUCCESS !=
        m_qnnFunctionPointers.qnnSystemInterface.systemContextCreate(
            &sysCtxHandle)) {
      QNN_ERROR("Could not create system handle.");
      returnStatus = StatusCode::FAILURE;
    }

    const QnnSystemContext_BinaryInfo_t *binaryInfo{nullptr};
    Qnn_ContextBinarySize_t binaryInfoSize{0};

    void *nonConstBuffer =
        const_cast<void *>(static_cast<const void *>(buffer));

    if (StatusCode::SUCCESS == returnStatus &&
        QNN_SUCCESS !=
            m_qnnFunctionPointers.qnnSystemInterface.systemContextGetBinaryInfo(
                sysCtxHandle, nonConstBuffer, bufferSize, &binaryInfo,
                &binaryInfoSize)) {
      QNN_ERROR("Failed to get context binary info");
      returnStatus = StatusCode::FAILURE;
    }

    if (StatusCode::SUCCESS == returnStatus &&
        !copyMetadataToGraphsInfo(binaryInfo, m_graphsInfo, m_graphsCount)) {
      QNN_ERROR("Failed to copy metadata.");
      returnStatus = StatusCode::FAILURE;
    }

    m_qnnFunctionPointers.qnnSystemInterface.systemContextFree(sysCtxHandle);
    sysCtxHandle = nullptr;

    if (StatusCode::SUCCESS == returnStatus &&
        nullptr == m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary) {
      QNN_ERROR("contextCreateFromBinaryFnHandle is nullptr.");
      returnStatus = StatusCode::FAILURE;
    }

    if (StatusCode::SUCCESS == returnStatus &&
        m_qnnFunctionPointers.qnnInterface.contextCreateFromBinary(
            m_backendHandle, m_deviceHandle,
            (const QnnContext_Config_t **)m_contextConfig, nonConstBuffer,
            bufferSize, &m_context, m_profileBackendHandle)) {
      QNN_ERROR("Could not create context from binary.");
      returnStatus = StatusCode::FAILURE;
    }

    if (ProfilingLevel::OFF != m_profilingLevel) {
      extractBackendProfilingInfo(m_profileBackendHandle);
    }

    m_isContextCreated = true;

    if (StatusCode::SUCCESS == returnStatus) {
      for (size_t graphIdx = 0; graphIdx < m_graphsCount; graphIdx++) {
        if (nullptr == m_qnnFunctionPointers.qnnInterface.graphRetrieve) {
          QNN_ERROR("graphRetrieveFnHandle is nullptr.");
          returnStatus = StatusCode::FAILURE;
          break;
        }
        if (QNN_SUCCESS != m_qnnFunctionPointers.qnnInterface.graphRetrieve(
                               m_context, (*m_graphsInfo)[graphIdx].graphName,
                               &((*m_graphsInfo)[graphIdx].graph))) {
          QNN_ERROR("Unable to retrieve graph handle for graph Idx: %d",
                    graphIdx);
          returnStatus = StatusCode::FAILURE;
        }
      }
    }

    if (StatusCode::SUCCESS != returnStatus) {
      QNN_DEBUG("Cleaning up graph Info structures.");
      qnn_wrapper_api::freeGraphsInfo(&m_graphsInfo, m_graphsCount);
    }

    return returnStatus;
  }

 private:
  // Backing storage for the spill-fill group-registration context config.
  // These must outlive the contextCreateFromBinary call, so they live as
  // members rather than locals in setSpillFillGroup().
  QnnHtpContext_CustomConfig_t m_sfHtpConfig{};
  QnnContext_Config_t m_sfCtxConfig{};
  QnnContext_Config_t *m_sfCtxConfigPtrs[2] = {nullptr, nullptr};
};

#endif  // QNNMODEL_HPP