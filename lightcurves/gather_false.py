import lightkurve as lk
from pathlib import Path

# Optional: folder to save FITS files
outdir = Path("./lc_examples")
outdir.mkdir(exist_ok=True)

# ✅ Modern way to search and download a light curve
search = lk.search_lightcurve("KOI-5715.01", mission="Kepler")
lcfile = search[0].download(download_dir=str(outdir))

# ✅ Extract the PDCSAP_FLUX as a LightCurve object
lc = lcfile.PDCSAP_FLUX  # attribute, not function

# Plot the light curve
lc.plot()
