param(
    [string]$PagesDir = "c:\Users\ysj\Desktop\upload_all\AI_edu_dataset\intermediate_results\pages",
    [string]$OutJson = "c:\Users\ysj\Desktop\upload_all\AI_edu_dataset\intermediate_results\ocr_all_pages.json"
)

Add-Type -AssemblyName System.Runtime.WindowsRuntime

$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1'
})[0]

function Await($WinRtTask, $ResultType) {
    $asTask = $asTaskGeneric.MakeGenericMethod($ResultType)
    $netTask = $asTask.Invoke($null, @($WinRtTask))
    $netTask.Wait(-1) | Out-Null
    $netTask.Result
}

[Windows.Storage.StorageFile, Windows.Storage, ContentType=WindowsRuntime] | Out-Null
[Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics.Imaging, ContentType=WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.SoftwareBitmap, Windows.Graphics.Imaging, ContentType=WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrEngine, Windows.Media.Ocr, ContentType=WindowsRuntime] | Out-Null
[Windows.Globalization.Language, Windows.Globalization, ContentType=WindowsRuntime] | Out-Null

$lang = [Windows.Globalization.Language]::new("ko")
$ocrEngine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage($lang)

$files = Get-ChildItem -Path $PagesDir -Filter "page_*.png" | Sort-Object Name
$results = @()
$sw = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($f in $files) {
    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($f.FullName)) ([Windows.Storage.StorageFile])
    $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $ocrRes = Await ($ocrEngine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
    $lines = @()
    foreach ($line in $ocrRes.Lines) {
        $lines += $line.Text
    }
    $results += [PSCustomObject]@{
        file = $f.Name
        page = [int]($f.BaseName -replace 'page_', '')
        text = ($lines -join "`n")
        lines = $lines
    }
    $stream.Dispose()
}

$sw.Stop()
$json = $results | ConvertTo-Json -Depth 4
[System.IO.File]::WriteAllText($OutJson, $json, [System.Text.UTF8Encoding]::new($false))
Write-Output ("OCR completed for {0} pages in {1:N2}s" -f $results.Count, $sw.Elapsed.TotalSeconds)
