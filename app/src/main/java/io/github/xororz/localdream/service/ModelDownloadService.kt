package io.github.xororz.localdream.service

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import io.github.xororz.localdream.R
import io.github.xororz.localdream.data.Model
import io.github.xororz.localdream.utils.Http
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit
import java.util.zip.ZipInputStream
import kotlin.coroutines.cancellation.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancel
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import okhttp3.Request

/** One file in a model manifest: a plain name and the size it must end up. */
private data class ManifestEntry(val name: String, val size: Long)

class ModelDownloadService : Service() {
    private val serviceScope = CoroutineScope(Dispatchers.IO + Job())
    private var downloadJob: Job? = null

    private val notificationManager by lazy {
        getSystemService(NOTIFICATION_SERVICE) as NotificationManager
    }

    private val client = Http.client.newBuilder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .build()

    companion object {
        private const val TAG = "ModelDownloadService"
        private const val NOTIFICATION_CHANNEL_ID = "model_download_channel"
        private const val NOTIFICATION_ID = 2001

        private val _downloadState = MutableStateFlow<DownloadState>(DownloadState.Idle)
        val downloadState: StateFlow<DownloadState> = _downloadState

        const val ACTION_START_DOWNLOAD = "action_start_download"
        const val ACTION_CANCEL_DOWNLOAD = "action_cancel_download"

        const val EXTRA_MODEL_ID = "model_id"
        const val EXTRA_MODEL_NAME = "model_name"
        const val EXTRA_FILE_URL = "file_url"
        const val EXTRA_IS_ZIP = "is_zip"

        // Version stamp written into the model directory on a successful NPU
        // download, and checked by Model.needsModelUpgrade. Defaults to the
        // shared "v3"; a model whose weights are rebuilt passes its own, so
        // only that model is invalidated instead of every NPU model at once.
        const val EXTRA_VERSION_MARKER = "version_marker"

        // Bumped when the Z-Image DiT was rebuilt with head-chunked attention.
        // The previous graphs ask the device for a 2.02 GB context and no
        // protection domain is that large, so an install carrying them cannot
        // generate at all -- and, being complete, would never have refreshed
        // itself on the shared marker.
        const val ZIMAGE_MARKER = "zimage_attn5"
        const val EXTRA_IS_NPU = "is_npu"
        const val EXTRA_MODEL_TYPE = "model_type" // "sd" or "upscaler"

        // A file URL ending in this is a list of files to fetch, not a model.
        const val MANIFEST_SUFFIX = "manifest.json"
    }

    sealed class DownloadState {
        object Idle : DownloadState()
        data class Downloading(
            val modelId: String,
            val progress: Float,
            val downloadedBytes: Long,
            val totalBytes: Long,
        ) : DownloadState()

        data class Extracting(val modelId: String) : DownloadState()
        data class Success(val modelId: String) : DownloadState()
        data class Error(val modelId: String, val message: String) : DownloadState()
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_DOWNLOAD -> {
                val modelId = intent.getStringExtra(EXTRA_MODEL_ID) ?: return START_NOT_STICKY
                val modelName = intent.getStringExtra(EXTRA_MODEL_NAME) ?: modelId
                val fileUrl = intent.getStringExtra(EXTRA_FILE_URL) ?: return START_NOT_STICKY
                val isZip = intent.getBooleanExtra(EXTRA_IS_ZIP, false)
                val isNpu = intent.getBooleanExtra(EXTRA_IS_NPU, false)
                val marker = intent.getStringExtra(EXTRA_VERSION_MARKER) ?: "v3"
                val modelType = intent.getStringExtra(EXTRA_MODEL_TYPE) ?: "sd"

                startForeground(NOTIFICATION_ID, createNotification(modelName, 0f))
                startDownload(modelId, modelName, fileUrl, isZip, isNpu, marker, modelType)
            }

            ACTION_CANCEL_DOWNLOAD -> {
                cancelDownload()
            }
        }
        return START_NOT_STICKY
    }

    private fun startDownload(
        modelId: String,
        modelName: String,
        fileUrl: String,
        isZip: Boolean,
        isNpu: Boolean,
        marker: String,
        modelType: String,
    ) {
        downloadJob?.cancel()
        downloadJob = serviceScope.launch {
            var tempFile: File? = null
            var extractTempDir: File? = null
            try {
                _downloadState.value = DownloadState.Downloading(modelId, 0f, 0, 0)

                val tempDir = File(filesDir, "temp_downloads")

                if (tempDir.exists()) {
                    tempDir.deleteRecursively()
                }
                tempDir.mkdirs()

                // A manifest means the model is published as loose files rather
                // than one archive. That is the only workable shape once a model
                // is several GB: a single zip has to be downloaded AND extracted,
                // so the device needs twice the model's size free, and a dropped
                // connection costs the whole archive instead of one file.
                if (modelType == "sd" && fileUrl.endsWith(MANIFEST_SUFFIX)) {
                    downloadFromManifest(modelId, modelName, fileUrl, isNpu, marker)
                    tempDir.deleteRecursively()

                    _downloadState.value = DownloadState.Success(modelId)
                    updateNotification(modelName, 100f, true)
                    withContext(Dispatchers.Main) {
                        kotlinx.coroutines.delay(2000)
                        _downloadState.value = DownloadState.Idle
                        stopForeground(STOP_FOREGROUND_REMOVE)
                        stopSelf()
                    }
                    return@launch
                }

                tempFile = File(tempDir, "${modelId}_${System.currentTimeMillis()}.tmp")

                downloadFile(fileUrl, tempFile, modelId, modelName)

                when (modelType) {
                    "sd" -> {
                        if (isZip) {
                            val modelDir = File(getModelsDir(), modelId)

                            if (modelDir.exists()) {
                                modelDir.deleteRecursively()
                            }
                            modelDir.mkdirs()

                            extractTempDir = File(tempDir, "${modelId}_extract")
                            extractTempDir.mkdirs()

                            _downloadState.value = DownloadState.Extracting(modelId)
                            updateNotification(modelName, 0f, isExtracting = true)

                            unzipFile(tempFile, extractTempDir)

                            extractTempDir.listFiles()?.forEach { file ->
                                file.renameTo(File(modelDir, file.name))
                            }
                            extractTempDir.delete()
                            extractTempDir = null

                            if (isNpu) {
                                File(modelDir, "v3").createNewFile()
                                if (marker != "v3") File(modelDir, marker).createNewFile()
                            }
                        }
                    }

                    "upscaler" -> {
                        val upscalerDir = File(getModelsDir(), modelId).apply {
                            if (!exists()) mkdirs()
                        }
                        val targetFile = File(upscalerDir, Model.UPSCALER_FILE_NAME)

                        if (targetFile.exists()) {
                            targetFile.delete()
                        }

                        // Don't report success on a failed move: it would leave
                        // an empty model dir that the UI/loader can't use.
                        if (!tempFile.renameTo(targetFile)) {
                            tempFile.copyTo(targetFile, overwrite = true)
                        }
                    }
                }

                tempFile.delete()
                tempFile = null

                _downloadState.value = DownloadState.Success(modelId)
                updateNotification(modelName, 100f, true)

                withContext(Dispatchers.Main) {
                    kotlinx.coroutines.delay(2000)
                    _downloadState.value = DownloadState.Idle
                    stopForeground(STOP_FOREGROUND_REMOVE)
                    stopSelf()
                }
            } catch (e: CancellationException) {
                // Cancellation (service reclaimed, a new download started, or
                // explicit cancel) is not a download failure: re-throw so it is
                // not surfaced as an "Error" state. Emitting Error here is what
                // produced the spurious "Job was cancelled" snackbar that could
                // appear right after a successful download finished.
                throw e
            } catch (e: Exception) {
                Log.e(TAG, "Download failed", e)

                tempFile?.delete()
                extractTempDir?.deleteRecursively()

                _downloadState.value =
                    DownloadState.Error(modelId, e.message ?: getString(R.string.unknown_error))
                updateNotification(modelName, 0f, false, e.message)

                withContext(Dispatchers.Main) {
                    kotlinx.coroutines.delay(3000)
                    _downloadState.value = DownloadState.Idle
                    stopForeground(STOP_FOREGROUND_REMOVE)
                    stopSelf()
                }
            }
        }
    }

    /**
     * Downloads one file, reporting progress against a possibly larger whole.
     *
     * [doneBefore] and [grandTotal] let a multi-file download report a single
     * progress bar across every file instead of restarting at 0% for each one.
     * For a single-file download they are 0 and the file's own length.
     */
    private suspend fun downloadFile(
        url: String,
        destFile: File,
        modelId: String,
        modelName: String,
        doneBefore: Long = 0L,
        grandTotal: Long = 0L,
    ) = withContext(Dispatchers.IO) {
        val request = Request.Builder()
            .url(url)
            .build()

        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw Exception(getString(R.string.error_download_failed, response.code.toString()))
            }

            val body = response.body ?: throw Exception("Response body is null")
            val totalBytes = body.contentLength()
            var downloadedBytes = 0L
            var lastUpdateTime = 0L

            java.io.BufferedOutputStream(FileOutputStream(destFile)).use { output ->
                body.byteStream().buffered().use { input ->
                    val buffer = ByteArray(32 * 1024)
                    var bytes: Int

                    while (input.read(buffer).also { bytes = it } != -1) {
                        output.write(buffer, 0, bytes)
                        downloadedBytes += bytes

                        val currentTime = System.currentTimeMillis()
                        if (currentTime - lastUpdateTime >= 500 || downloadedBytes == totalBytes) {
                            lastUpdateTime = currentTime
                            val shownDone = doneBefore + downloadedBytes
                            val shownTotal =
                                if (grandTotal > 0) grandTotal else totalBytes
                            val progress = if (shownTotal > 0) {
                                shownDone.toFloat() / shownTotal
                            } else {
                                0f
                            }

                            _downloadState.value = DownloadState.Downloading(
                                modelId,
                                progress,
                                shownDone,
                                shownTotal,
                            )

                            updateNotification(modelName, progress)
                        }
                    }
                }
            }

            // Guard against silently truncated downloads: a dropped connection
            // ends the read loop without throwing, leaving a partial file.
            if (totalBytes > 0 && downloadedBytes != totalBytes) {
                throw Exception(
                    getString(R.string.error_download_failed, "$downloadedBytes/$totalBytes"),
                )
            }
        }
    }

    /**
     * Downloads a model published as loose files listed in a manifest.
     *
     *   { "files": [ { "name": "unet_part1.bin", "size": 384131072 }, ... ] }
     *
     * Names are resolved against the manifest's own directory, so the manifest
     * and the files it lists live side by side and the whole model can be moved
     * or mirrored by copying one directory.
     *
     * Each file is written to `<name>.part` and renamed only once its length
     * matches the manifest. That makes the download resumable at file
     * granularity: a run that dies partway leaves the finished files in place,
     * and a retry re-fetches only what is missing or the wrong size. It also
     * means a truncated file can never be mistaken for a complete one, which
     * matters because the backend mmaps these and would read off the end.
     */
    private suspend fun downloadFromManifest(
        modelId: String,
        modelName: String,
        manifestUrl: String,
        isNpu: Boolean,
        marker: String,
    ) = withContext(Dispatchers.IO) {
        val baseUrl = manifestUrl.substringBeforeLast('/', "")
        if (baseUrl.isEmpty()) throw Exception("Bad manifest URL: $manifestUrl")

        val manifestJson = client.newCall(Request.Builder().url(manifestUrl).build())
            .execute().use { response ->
                if (!response.isSuccessful) {
                    throw Exception(
                        getString(R.string.error_download_failed, response.code.toString()),
                    )
                }
                response.body?.string() ?: throw Exception("Empty manifest")
            }

        val entries = org.json.JSONObject(manifestJson).getJSONArray("files")
        if (entries.length() == 0) throw Exception("Manifest lists no files")

        val files = (0 until entries.length()).map { i ->
            val o = entries.getJSONObject(i)
            val name = o.getString("name")
            // Names come off the network; keep them to plain filenames so a
            // crafted manifest cannot write outside the model directory.
            if (name.contains('/') || name.contains('\\') || name == "." || name == "..") {
                throw Exception("Manifest entry is not a plain file name: $name")
            }
            ManifestEntry(name, o.optLong("size", -1L))
        }

        // Download into a staging directory, not the model directory.
        //
        // Model.isModelDownloaded treats any non-empty model directory as a
        // complete model. Writing files there as they arrive would make an
        // interrupted download indistinguishable from a finished one: the app
        // would list the model as ready, the backend would fail on the parts
        // that never arrived, and there would be no download button left to
        // retry with. Staging keeps that state invisible until every file is
        // present, while still persisting across attempts so a retry resumes
        // instead of starting over.
        val modelDir = File(getModelsDir(), modelId)
        val stageDir = File(getModelsDir(), ".staging_$modelId").apply { mkdirs() }

        // A previous attempt may have staged files for a different manifest --
        // a re-converted model with a different part count, say. Leaving them
        // would mix parts from two conversions, and the backend discovers parts
        // by counting upward from 1, so the extras would be loaded as if they
        // belonged.
        val wanted = files.map { it.name }.toSet()
        stageDir.listFiles()?.forEach { f ->
            if (f.name !in wanted && !f.name.endsWith(".part")) {
                Log.i(TAG, "manifest: dropping stale staged " + f.name)
                f.delete()
            }
        }

        val grandTotal = files.sumOf { if (it.size > 0) it.size else 0L }

        // Anything already the right size is left alone, so a retry resumes.
        var done = files.filter { e ->
            e.size > 0 && File(stageDir, e.name).length() == e.size
        }.sumOf { it.size }

        for (entry in files) {
            ensureActive()
            val target = File(stageDir, entry.name)
            if (entry.size > 0 && target.length() == entry.size) {
                Log.i(TAG, "manifest: ${entry.name} already complete, skipping")
                continue
            }
            target.delete()

            val part = File(stageDir, entry.name + ".part")
            part.delete()
            downloadFile("$baseUrl/${entry.name}", part, modelId, modelName, done, grandTotal)

            if (entry.size > 0 && part.length() != entry.size) {
                part.delete()
                throw Exception(
                    getString(
                        R.string.error_download_failed,
                        "${entry.name} ${part.length()}/${entry.size}",
                    ),
                )
            }
            if (!part.renameTo(target)) {
                part.delete()
                throw Exception("Could not finalize ${entry.name}")
            }
            done += if (entry.size > 0) entry.size else target.length()
        }

        // Both stamps: "v3" keeps the shared NPU check satisfied, and the
        // per-model marker is what lets this model alone be invalidated the
        // next time its weights are rebuilt.
        if (isNpu) {
            File(stageDir, "v3").createNewFile()
            if (marker != "v3") File(stageDir, marker).createNewFile()
        }

        // Everything is present: swap staging into place as one step. Any
        // earlier install is removed first so a re-download cannot leave files
        // from the previous manifest behind.
        if (modelDir.exists() && !modelDir.deleteRecursively()) {
            throw Exception("Could not replace the existing model directory")
        }
        if (!stageDir.renameTo(modelDir)) {
            throw Exception("Could not move the downloaded model into place")
        }
    }

    private suspend fun unzipFile(zipFile: File, destDir: File) = withContext(Dispatchers.IO) {
        ZipInputStream(zipFile.inputStream().buffered()).use { zis ->
            var entry = zis.nextEntry

            while (entry != null) {
                if (!entry.isDirectory) {
                    val fileName = entry.name.substringAfterLast('/')
                    if (fileName.isNotEmpty() && !fileName.startsWith(".") && !fileName.startsWith("__MACOSX")) {
                        val file = File(destDir, fileName)

                        java.io.BufferedOutputStream(FileOutputStream(file)).use { output ->
                            zis.copyTo(output)
                        }
                    }
                }
                zis.closeEntry()
                entry = zis.nextEntry
            }
        }
    }

    private fun cancelDownload() {
        downloadJob?.cancel()
        _downloadState.value = DownloadState.Idle
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun getModelsDir(): File = File(filesDir, "models").apply {
        if (!exists()) mkdirs()
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            NOTIFICATION_CHANNEL_ID,
            getString(R.string.model_download_channel),
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = getString(R.string.model_download_channel_desc)
        }
        notificationManager.createNotificationChannel(channel)
    }

    private fun createNotification(
        modelName: String,
        progress: Float,
        isExtracting: Boolean = false,
    ): android.app.Notification {
        val title = if (isExtracting) {
            getString(R.string.extracting)
        } else {
            getString(R.string.downloading_model, modelName)
        }

        val openAppIntent = packageManager.getLaunchIntentForPackage(packageName)?.apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_NEW_TASK
        }
        val appPendingIntent = PendingIntent.getActivity(
            this,
            0,
            openAppIntent,
            PendingIntent.FLAG_IMMUTABLE,
        )

        return NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
            .setContentTitle(title)
            .setSmallIcon(android.R.drawable.stat_sys_download)
            .setProgress(100, (progress * 100).toInt(), isExtracting)
            .setOngoing(true)
            .setContentIntent(appPendingIntent)
            .build()
    }

    private fun updateNotification(
        modelName: String,
        progress: Float,
        success: Boolean = false,
        error: String? = null,
        isExtracting: Boolean = false,
    ) {
        val notification = when {
            success -> {
                NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
                    .setContentTitle(getString(R.string.download_complete))
                    .setContentText(modelName)
                    .setSmallIcon(android.R.drawable.stat_sys_download_done)
                    .setOngoing(false)
                    .build()
            }

            error != null -> {
                NotificationCompat.Builder(this, NOTIFICATION_CHANNEL_ID)
                    .setContentTitle(getString(R.string.download_failed))
                    .setContentText(error)
                    .setSmallIcon(android.R.drawable.stat_notify_error)
                    .setOngoing(false)
                    .build()
            }

            else -> {
                createNotification(modelName, progress, isExtracting)
            }
        }

        notificationManager.notify(NOTIFICATION_ID, notification)
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onTimeout(startId: Int) {
        super.onTimeout(startId)
        handleTimeout(0)
    }

    override fun onTimeout(startId: Int, fgsType: Int) {
        super.onTimeout(startId, fgsType)
        handleTimeout(fgsType)
    }

    private fun handleTimeout(fgsType: Int) {
        Log.e(TAG, "Foreground service timeout (fgsType=$fgsType)")
        downloadJob?.cancel()
        _downloadState.value = DownloadState.Error("timeout", "Foreground service timeout")
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
    }
}
