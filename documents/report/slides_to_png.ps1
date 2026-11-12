# Export every slide of a deck to PNG through PowerPoint, and also save a PDF copy next to the deck.
# Usage: powershell -ExecutionPolicy Bypass -File slides_to_png.ps1 <deck.pptx> <output-folder>
param([string]$Deck, [string]$Folder)
$full = (Resolve-Path $Deck).Path
New-Item -ItemType Directory -Force $Folder | Out-Null
Get-ChildItem $Folder -Filter "slide-*.png" | Remove-Item -Force
$app = New-Object -ComObject PowerPoint.Application
try {
    # ReadOnly, Untitled, WithWindow = false
    $pres = $app.Presentations.Open($full, -1, 0, 0)
    $count = $pres.Slides.Count
    for ($i = 1; $i -le $count; $i++) {
        $pres.Slides.Item($i).Export((Join-Path $Folder ("slide-{0}.png" -f $i)), "PNG", 1600, 900)
    }
    $pres.SaveAs([System.IO.Path]::ChangeExtension($full, ".pdf"), 32)
    "Slides: $count"
    $pres.Close()
}
finally {
    $app.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
