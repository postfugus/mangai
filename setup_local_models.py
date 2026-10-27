"""Download the local French OCR and offline French→Russian model files once."""
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / ".local_models"
PACKAGES = {
    "fr_en": "https://argos-net.com/v1/translate-fr_en-1_9.argosmodel",
    "en_ru": "https://argos-net.com/v1/translate-en_ru-1_9.argosmodel",
}
FRENCH_OCR = (
    "https://raw.githubusercontent.com/tesseract-ocr/tessdata_fast/main/"
    "fra.traineddata"
)


def download(url: str, destination: Path):
    if destination.is_file() and destination.stat().st_size > 1024:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {destination.name} …")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)


def install_argos_package(code: str, url: str):
    package_dir = MODEL_ROOT / code
    if (package_dir / "model").is_dir() and (package_dir / "sentencepiece.model").is_file():
        return
    archive_path = MODEL_ROOT / f"{code}.argosmodel"
    download(url, archive_path)
    package_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            relative = PurePosixPath(member.filename)
            # Model files are stored below one package-name directory.
            parts = relative.parts[1:]
            if not parts or parts[0] not in {"metadata.json", "sentencepiece.model", "model"}:
                continue
            target = package_dir.joinpath(*parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                output.write(source.read())
    archive_path.unlink(missing_ok=True)


def main():
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    (MODEL_ROOT / "tessdata").mkdir(parents=True, exist_ok=True)
    download(FRENCH_OCR, MODEL_ROOT / "tessdata" / "fra.traineddata")
    for code, url in PACKAGES.items():
        install_argos_package(code, url)
    print("Local OCR and French→Russian translation models are ready.")


if __name__ == "__main__":
    main()
