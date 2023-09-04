import numpy as np
from scipy.ndimage import gaussian_filter
import xarray as xr

# Provided by the beamline team. Used for accessing the data from the hRIXS instrument
# Documentation at: https://rtd.xfel.eu/docs/scs-toolbox/en/latest/HRIXS.html
import toolbox_scs as tb

################################################################
# Run and image information
################################################################

def energy_calibration(x):
    """ Calibration of the detector from pixel along dispersive axis to photon energy in eV"""
    calib = 0.105107
    inters = 879.47183
    return x * calib + inters

# manual list of dark runs for fallback
dark_runs = np.array((40, 122, 141, 148, 149, 193, 208, 215, 261, 317, 387, 424, 447, 448, 450, 467, 472, 538, 594))

# list of the xgm calibration runs, in pairs of (transmission [%], run number)
# runs without transmission numbers: 422, 426, 320
xgm_calib_runs = np.array((
    (100, 156), (50, 157), (30, 158), (30, 159), (20, 160), (10, 161), 
    (1.67, 163), (0.6, 164), (0.28, 165), (50, 184), (0.17, 256), (3.4, 368)
))

################################################################
# Data handling class
################################################################

class HrixsData:
    """ Class for data analysis of data from the hRIXS experiment.

    Many properties rely on the main dataset of the object to save the result
    and will only recalculate, if the result is missing in the dataset.

    Instances should generally be obtained using the get_run method.
    It reuses previously generated instances, such that calculations
    do not have to be redone.
    If the data should be modified and these modiufications change
    the calculated data, a new object should be created manually. This
    will overwrite the old object.
    """
    proposal: int = 3154
    
    _instances = {}
    
    @classmethod
    def get_run(cls, run_number: int):
        """ Returns the instance to the run_number or initialises
        a new object if it does not already exist
        """
        if run_number in cls._instances:
            return cls._instances[run_number]
        return cls(run_number)
    
    def _register_run(self, run_number):
        self._instances[run_number] = self
    
    def __init__(self, run_number: int):
        self.run_number = run_number
        self.run, self.ds = tb.load(self.proposal, run_number, fields=["hRIXS_det", "transmission"])
        
        self.ds = self.ds.assign_coords(
            x = self.ds.x.values,
            y = self.ds.y.values,
            energy = ('y', energy_calibration(self.ds.y.values))
        )
        
        # The detector has some bad indices, this finds those
        # Can not use the self.mean property, as the hRIXS_det is modified afterwards
        self.ds["bad_y"] = xr.where(
            # The threshold for the bad indeces can be adjusted here
            self.ds.hRIXS_det.mean("trainId").max("x") > 10000,
            True,
            False
        ).astype(bool)
        
        self.ds.hRIXS_det.loc[dict(y=self.bad_indices)] = 0
        
        self._register_run(run_number)
        
        self._normalized_raw_beam = None
       
    def get_closest_run_number(self, run_numbers: list[int]):
        """ Returns the closest run number from the given list of run_numbers """
        return int(run_numbers[abs(run_numbers - self.run_number).argmin()])
    
    ################################################################
    # Beam properties
    ################################################################
    
    @property
    def num_pulses(self):
        if not "num_pulses" in self.ds.keys():
            xgm_scs = tb.get_array(self.run, 'SCS_SA3')
            self.ds["num_pulses"] = xgm_scs.where(xgm_scs != 1, drop=True).sizes["XGMbunchId"]
        return self.ds.num_pulses
    
    @property
    def sample_transmission(self):
        if not "sample_transmission" in self.ds.keys():
            self.ds["sample_transmission"] = self.ds.transmission.mean()
        return self.ds.sample_transmission

    @property
    def beam_intensity(self):
        if not "beam_intensity" in self.ds.keys():
            def load_xgm_array(run, ds, array_name):
                xgm_arr = tb.get_array(run, array_name)
                xgm_arr = xgm_arr.where(xgm_arr != 1., drop=True).rename(XGMbunchId='sa3_pId')
                return ds.merge(xgm_arr)
            
            calib_transmission = xgm_calib_runs[abs(xgm_calib_runs[:, 0] - self.sample_transmission.data).argmin(), 0]
            xgm_run_number = self.get_closest_run_number(xgm_calib_runs[xgm_calib_runs[:, 0] == calib_transmission, 1])
            self.ds["used_xgm_calibration_run"] = xgm_run_number
            
            dtype = [('XGM_run', 'i'), ('XGM_used', 'i'), ('calib', 'f')]
            # Loads the XGM calibration.
            # always use XTD10, as SCS seems to be unreliable
            data = np.loadtxt('XGM_calibration_XTD10.txt', dtype=dtype)
            idx = np.argmin(np.abs(data['XGM_run']-xgm_run_number))
            
            if data['XGM_used'][idx]==0:# use XTD10
                self.ds["used_xgm"] = "XTD10_SA3"
                dat_ds = xr.Dataset()
                dat_ds = load_xgm_array(self.run, dat_ds, "XTD10_SA3")
                xgm_arr = dat_ds.XTD10_SA3 * data['calib'][idx] * (self.ds['transmission'].mean()*0.01) * 0.33
            else:# use SCS
                self.ds["used_xgm"] = "SCS_SA3"
                dat_ds = xr.Dataset()
                dat_ds = load_xgm_array(self.run, dat_ds, "SCS_SA3")
                xgm_arr = dat_ds.SCS_SA3 * data['calib'][idx]
                
            self.ds["beam_intensity"] = xgm_arr.mean()
        return self.ds.beam_intensity
    
    @property
    def normalized_raw_beam(self):
        """ Returns the FEL spectrum """
        if self._normalized_raw_beam is None:
            energy, spectrum = np.loadtxt("fel spectra/run0080.txt", unpack=True, usecols=(0, 1))
            energy, spectrum = energy[::-1], spectrum[::-1]
            spectrum_area = np.trapz(spectrum, energy)
            spectrum *= self.beam_intensity.data/spectrum_area
            
            self._normalized_raw_beam = energy, spectrum
        return self._normalized_raw_beam
    
    @property
    def beam_spectrum(self):
        if not "beam_spectrum" in self.ds.keys():
            self.ds["beam_spectrum"] = "y", self.get_beam_spectrum_offset(0)
        return self.ds.beam_spectrum

    def get_beam_spectrum_offset(self, offset, energy_eV = None):
        """ Returns the beam spectrum interpolated over a given photon energy range.
        The spectrum can be shifted using offset. Setting energy_eV to None (default)
        will use the energy sample points of the run.
        """
        if energy_eV is None:
            energy_eV = self.ds.energy
        energy, spectrum = self.normalized_raw_beam
        return np.interp(energy_eV - offset, energy, spectrum, left=0, right=0)
        
    
    ################################################################
    # Main image
    ################################################################
    
    @property
    def bad_indices(self):
        return self.ds.where(self.ds.bad_y, drop=True).y
    
    @property
    def mean(self):
        """ Mean image of the run """
        if not "hRIXS_mean" in self.ds.keys():
            self.ds["hRIXS_mean"] = self.ds.hRIXS_det.mean("trainId")
        return self.ds.hRIXS_mean
    
    @property
    def blurred(self):
        """ A blurred image of the run. Used to find the maximum. """
        if not "hRIXS_blurred" in self.ds.keys():
            self.ds["hRIXS_blurred"] = ("x", "y"), gaussian_filter(self.mean, sigma=3)
        return self.ds.hRIXS_blurred
    
    @property
    def exposure_time(self):
        """ Returns the image exposure time in seconds. """
        if not "exposure_time" in self.ds.keys():
            exposure_time = tb.get_array(self.run, "hRIXS_exposure")
            self.ds["exposure_time"] = 1e-3 * exposure_time.mean()
        return self.ds.exposure_time
    
    ################################################################
    # ROI
    ################################################################
    
    def _get_sum_in_rect(self, x, y, rx, ry):
        return self.mean.sel(
            x = slice(x-rx, x+rx),
            y = slice(y-ry, y+ry)
        ).sum()
    
    def _find_radius(self, axis, center, r_init=(2, 2), dr=(10, 2), ratio=(0.1, 0.1)):
        r = [0, 0]
        axis_index = 0
        if axis == "y":
            axis_index = 1

        r[1 - axis_index] = r_init[1 - axis_index]
        last_included_sum = self._get_sum_in_rect(*center, *r)
        r[axis_index] = r_init[axis_index]

        running_ratio = 1

        while running_ratio > ratio[axis_index]:
            included_sum = self._get_sum_in_rect(*center, *r)
            running_ratio = included_sum / last_included_sum - 1
            last_included_sum = included_sum

            r[axis_index] += dr[axis_index]

        return r
    
    @property
    def roi(self):
        """ The region of interest (ROI) of the run.
        Uses a box around the maximum of the image. The box is expanded, until
        the resulting increase of the sum levels.
        """
        if not "roi" in self.ds.keys():
            # Use the blurred image to supress single extreme pixels from fluctuations
            maximum = self.mean.sel(
                self.blurred.where(
                    (self.blurred.x > 500) & (self.blurred.x < 1500)
                    & (self.blurred.y > 250) & (self.blurred.y < 1000),
                    0
                ).argmax(("x", "y"))
            )
            
            radius = self._find_radius("x", (maximum.x, maximum.y))
            radius = self._find_radius("y", (maximum.x, maximum.y), radius)

            self.ds["roi"] = xr.where(
                (self.ds.x > maximum.x - radius[0]) & (self.ds.x < maximum.x + radius[0])
                & (self.ds.y > maximum.y - radius[1]) & (self.ds.y < maximum.y + radius[1]),
                True,
                False
            ).astype(bool)
        
        return self.ds.roi
    
    @roi.setter
    def roi(self, new_roi):
        self.ds["roi"] = new_roi
        
        # Remove any variables in the dataset that depend on the ROI
        self.ds = self.ds.drop_vars(
            [
                "sum_roi",
                "is_data_image",
                "image_data",
                "image_dark",
                "image_noBG"
            ],
            errors = "ignore"
        )

    def select_roi(self, da, axis="both", radius=0):
        """ Removes any data from a dataarray from this run outside of the ROI.
        Axis defines the considered axis (if applicable).
        Radius defines an additional radius in pixels around the ROI to be included.
        """
        selection = {}
        
        if axis == "both":
            if "x" in da.dims:
                selection["x"] = None
            if "y" in da.dims:
                selection["y"] = None
        else:
            selection[axis] = None
        
        roi_region = self.roi.where(self.roi, drop=True)
        for key in selection.keys():
            selection[key] = slice(
                roi_region[key].min() - radius,
                roi_region[key].max() + radius
            )
        
        return da.sel(selection)
    
    ################################################################
    # Image filtering
    ################################################################
    
    @property
    def detector_images(self):
        """ All images in the run """
        return self.ds.hRIXS_det
    
    @property
    def is_data_image(self):
        """ Boolean array, whether an image has a data peak.
        The mean over the ROI is copmpared to the standard deviation.
        An image with a mean > 5 strandard deviations is assumed to have data.
        """
        if not "is_data_image" in self.ds.keys():
            external_dark_image = self.get_run(self.get_closest_run_number(dark_runs)).mean
            noBG_images = self.select_roi(self.detector_images - external_dark_image, axis="x")
            std_dev = noBG_images.where((890 <= noBG_images.energy) & (noBG_images.energy <= 910), drop=True).mean("x").std("y").mean("trainId")

            self.ds["std_dev"] = std_dev
            self.ds["roi_mean"] = self.select_roi(noBG_images, axis="y").mean(("x", "y"))
            self.ds["is_data_image"] = self.ds.roi_mean > std_dev * 5
            
        return self.ds.is_data_image
    
    @property
    def data(self):
        """ Mean over all the images with a data peak. """
        if not "image_data" in self.ds.keys():
            self.ds["image_data"] = self.detector_images.where(self.is_data_image, drop=True).mean("trainId")
        return self.ds.image_data
    
    @property
    def dark(self):
        """ Mean over all the images without a data peak. If less than 10 images have a data
        peak, the closest dark runs images are also used.
        """
        if not "image_dark" in self.ds.keys():
            if self.is_data_image.where(self.is_data_image == False, drop=True).count() < 10:
                # if all images are data images, take the closest dark run to the actual run and use that
                dark_run_number = self.get_closest_run_number(dark_runs)
                print(f"run {self.run_number} does not have enough dark images, additionally using dark run {dark_run_number}")
                self.ds["image_dark"] = xr.concat(
                    (
                        self.get_run(dark_run_number).detector_images,
                        self.detector_images.where(self.is_data_image == False, drop=True)
                    ),
                    "trainId"
                ).mean("trainId")
            else:
                self.ds["image_dark"] = self.detector_images.where(self.is_data_image == False, drop=True).mean("trainId")
        return self.ds.image_dark
    
    @property
    def noBG(self):
        """ Image without Background, ie noBG = data - dark"""
        if not "image_noBG" in self.ds.keys():
            self.ds["image_noBG"] = self.data - self.dark
        return self.ds.image_noBG
    
    ################################################################
    # Data
    ################################################################
    
    def get_raw_spectrum(self, radius=None, image=None):
        """ Returns the spectrum of the given image by summing over the
        non-dispersive axis. Radius is used to define the summed region by 
        specifying a radius around the ROI. By radius to None (default), the
        whole image is used. Image specifies the used image. By setting image
        to None (default), the noBG image is used."""
        if image is None:
            image = self.noBG

        if radius is None:
            spectrum = image.sum("x")
        else:
            spectrum = self.select_roi(image, axis="x", radius=radius).sum("x")

        # for the offset, select a region without signal
        offset = spectrum.where((890 <= spectrum.energy) & (spectrum.energy <= 910), drop=True).mean()
        return (spectrum - offset)
    
    def get_spectrum(self, radius=None, image=None):
        """ Returns the normalised spectrum by deviding the detector counts
        by the total beam energy.
        """
        return self.get_raw_spectrum(radius, image) / ( self.beam_intensity * self.num_pulses * self.exposure_time * 10 )

    def get_spectrum_deviation(self, radius=100):
        """ Returns the standard deviation of the spectrum across all data images. """
        return self.get_spectrum(
            radius,
            self.detector_images.where(self.is_data_image, drop=True)
        ).std("trainId")
    
    def export_spectrum(self, min_energy=None, max_energy=None, roi_radius=100, filename=None):
        if min_energy is None:
            min_energy = self.ds.energy.min()
        if max_energy is None:
            max_energy = self.ds.energy.max()
        
        if filename is None:
            filename = f"spectrum exports/run_{self.run_number}.dat"
        
        spectrum = self.get_spectrum(roi_radius)
        spectrum = spectrum.where((spectrum.energy > min_energy) & (spectrum.energy < max_energy), drop=True)
        
        deviation = self.get_spectrum(
            roi_radius,
            self.detector_images.where(self.is_data_image, drop=True)
        ).std("trainId")
        deviation = deviation.where((deviation.energy > min_energy) & (deviation.energy < max_energy), drop=True)
        
        np.savetxt(
            filename,
            np.stack((spectrum.energy, spectrum.values, deviation.values), 1),
            header = "Energy [eV] - Intensity [a.u.] - Standard deviation [a.u.]"
        )
        