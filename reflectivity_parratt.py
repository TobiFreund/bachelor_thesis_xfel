import numpy as np
import scipy.constants as const

################################################################
# Material data
################################################################

class MaterialManager:
    atomic_masses = {
        "B": 10.81,
        "C": 12.01,
        "Si": 28.09,
        "Cu": 63.55
    }

    dirname_scattering = "scattering_factors"

    _material_scattering_cache = {}

    # Holds the element composition for each material:
    # {name: {element: amount}}
    _materials: dict[str, dict[str, int]] = {}

    @classmethod
    def add_material(cls, material_name: str, composition: dict[str, int]):
        cls._materials[material_name] = composition

    @classmethod
    def material_mass(cls, material_name: str):
        if not material_name in cls._materials:
            return None
        
        mass = 0
        for element, amount in cls._materials[material_name].items():
            mass += cls.atomic_masses[element] * amount
        
        return mass

    @classmethod
    def _scattering_list(cls, element_name: str):
        """ Returns the scattering factors as loaded from the corresponding file.
        Returns the tuple (f1, f2), where f1 and f2 each are a 2 dimensional array
        with rows of X-Ray energy - scattering factor pairs.
        """
        if not element_name in cls._material_scattering_cache:
            fname_ending = "ffast"
            if element_name == "Cu":
                fname_ending = "abs-KK-v2"
            f1_filename = f"{cls.dirname_scattering}/{element_name}-f1-{fname_ending}.dat"
            f2_filename = f"{cls.dirname_scattering}/{element_name}-f2-{fname_ending}.dat"
        
            f1 = np.loadtxt(f1_filename)
            f2 = np.loadtxt(f2_filename)
            cls._material_scattering_cache[element_name] = f1, f2
    
        return cls._material_scattering_cache[element_name]

    @classmethod
    def scattering_factor(cls, material_name: str, energy_eV: float):
        """ Calculates the complex scattering factor for the material. The energy can
        be a float or any numpy array of floats. Does an interpolation over the given
        datapoints.
        """
        if not material_name in cls._materials:
            return None
        
        f1 = 0
        f2 = 0
        for element, amount in cls._materials[material_name].items():
            f1_elm, f2_elm = cls._scattering_list(element)
            f1 += amount * np.interp(energy_eV * 10**-3, f1_elm[:, 0], f1_elm[:, 1])
            f2 += amount * np.interp(energy_eV * 10**-3, f2_elm[:, 0], f2_elm[:, 1])
        
        return f1 - 1j*f2

class SampleData:
    """ Handles the reading of the file with the sample information.
    """
    def __init__(self, filename):
        parameters = np.loadtxt(filename, dtype=str, skiprows=2, usecols=(0))
        values = np.loadtxt(filename, dtype=float, skiprows=2, usecols=(1))

        self._data = {p: v for p, v in zip(parameters, values)}

    def get_parameter(self, layer, parameter_name):
        """ Internal function to get a parameter from the file """
        return self._data[f"{layer}.set{parameter_name}"]

    def get_thickness(self, layer):
        """ Returns the thickness of the given layer in nm """
        # saved in angstrom
        return self.get_parameter(layer, "D") * 1e-1

    def get_roughness(self, layer):
        """ Returns the roughness of the given layer in nm """
        # saved in angstrom
        return self.get_parameter(layer, "Sigma") * 1e-1

    def get_density(self, layer):
        """ Returns the density of the given layer in kg/m^3 """
        # saved in hg/cm^3
        return self.get_parameter(layer, "Dens") * 1e5

    def get_layer_period(self):
        """ Returns the layer period of the sample in nm """
        # saved in angstrom
        return self.get_parameter("cp", "Period") * 1e-1


################################################################
# Multilayer model
################################################################

def scattering_eres(energy_eV, center_energy_eV, fERES = 18.9, gamma2 = 1.6):
    """ Calculates the adjustment scattering factor due to ERES effects """
    fadj = fERES * gamma2 * (
        ((energy_eV - center_energy_eV) - 1j*gamma2)
        / ((energy_eV - center_energy_eV)**2 + gamma2**2)
        )
    return fadj


class SampleLayer:
    def __init__(self, material: str):
        self._material = material

        self.thickness: float = 0
        self.roughness: float = 0
        self.density: float = 0

        self.use_ERES: bool = False

    @property
    def material(self):
        return self._material

    def susceptibility(self, energy_eV, fERES = 0, gamma2 = 0):
        Veff = MaterialManager.material_mass(self.material) \
            * const.atomic_mass / (self.density * 1e-27)
        f_layer = MaterialManager.scattering_factor(self.material, energy_eV)
        if self.use_ERES and self.material == "Cu":
            f_layer += scattering_eres(energy_eV, 929, fERES, gamma2)

        re = 1e9 * const.value('classical electron radius') # converted to nm
        wl_nm = 1e9 * const.h * const.c / energy_eV / const.eV

        return - ( re * wl_nm**2 / np.pi ) * f_layer / Veff


class SubMultilayer:
    def __init__(self, name: str, materials: list[str]):
        self._name = name
        
        self._layers: list[SampleLayer] = []
        for mat in materials:
            self._layers.append(SampleLayer(mat))
        
        self._layer_period = None

        # Parameter to control, whether the period or the thickness
        # is used for the last layer
        self.use_period = True

    @property
    def name(self):
        return self._name

    @property
    def materials(self):
        return [sl.material for sl in self._layers]

    @property
    def layer_count(self):
        return len(self._layers)
    
    def get_layer(self, layer) -> SampleLayer:
        """ Returns the corresponding layer. Can be indexed by position
        or name of the wanted layer.
        """
        if type(layer) == str:
            layer = self.materials.index(layer)
        
        return self._layers[layer]

    def init_values(self, sample_data: SampleData):
        """ Initializes the layers from the given SampleData object. """
        for sl in self._layers:
            layer_name = f"{self.name}_{sl.material}"
            sl.thickness = sample_data.get_thickness(layer_name)
            sl.roughness = sample_data.get_roughness(layer_name)
            sl.density = sample_data.get_density(layer_name)
        
        self._layer_period = sample_data.get_layer_period()
    
    def thickness_array(self):
        """ Stacks the thickness values of all layers into an array. """
        arr = np.array([sl.thickness for sl in self._layers])

        if self.use_period:
            arr[-1] = self._layer_period - arr[:-1].sum()
        return arr

    def roughness_array(self):
        """ Stacks the thickness values of all layers into an array. """
        return np.array([sl.roughness for sl in self._layers])

    def susceptibility_array(self, energy_eV, fERES = 0, gamma2 = 0):
        """ Stacks the susceptibility arrays of all layers.
        Arguments:
            energy_eV (np.ndarray, ndim=1): Array of the used energy values
            fERES (float): Parameter controlling the ERES effect
            gamma2 (float): Parameter controlling the ERES effect
        """
        energy_eV = np.array(energy_eV, ndmin=1)
        return np.stack([sl.susceptibility(energy_eV, fERES, gamma2) for sl in self._layers])


class MultilayerModel:
    def __init__(self, sub_multilayers: list[SubMultilayer], sample_data: SampleData):
        self._sub_multilayers = sub_multilayers
        self._repetitions = [1] * len(self._sub_multilayers)

        # The used values for the substrate
        self.substrate_thickness = np.array(0, ndmin=1)
        self.substrate_roughness = np.array(sample_data.get_roughness("Sub"), ndmin=1)
        self.substrate_susceptibility = np.array((-0.1033e-2 + 1j * 0.92017e-4), ndmin=1)

    def get_sub_multilayer(self, sub_multilayer) -> SubMultilayer:
        """ Returns the corresponding sub multilayer. Can be indexed by position
        or name of the wanted sub multilayer.
        """
        if type(sub_multilayer) == str:
            sub_multilayer = [sml.name for sml in self._sub_multilayers].index(sub_multilayer)
        
        return self._sub_multilayers[sub_multilayer]

    def set_multilayer_repetitions(self, sub_multilayer, repetitions: int):
        if type(sub_multilayer) == str:
            sub_multilayer = [sml.name for sml in self._sub_multilayers].index(sub_multilayer)
        
        self._repetitions[sub_multilayer] = repetitions
    
    def get_multilayer_repetitions(self, sub_multilayer) -> int:
        if type(sub_multilayer) == str:
            sub_multilayer = [sml.name for sml in self._sub_multilayers].index(sub_multilayer)
        
        return self._repetitions[sub_multilayer]

    def thickness_array(self):
        """ Stacks the thickness arrays of all sub multilayers. 
        Repeats the sub multilayers set by set_multilayer_repetitions.
        """
        return np.concatenate(
            [np.tile(sml.thickness_array(), rep) for sml, rep in zip(self._sub_multilayers, self._repetitions)]
            + [self.substrate_thickness]
        )

    def roughness_array(self):
        """ Stacks the roughness arrays of all sub multilayers. 
        Repeats the sub multilayers set by set_multilayer_repetitions.
        """
        return np.concatenate(
            [np.tile(sml.roughness_array(), rep) for sml, rep in zip(self._sub_multilayers, self._repetitions)]
            + [self.substrate_roughness]
        )

    def susceptibility_array(self, energy_eV, fERES = 0, gamma2 = 0):
        """ Stacks the susceptibility arrays of all sub multilayers. 
        Repeats the sub multilayers set by set_multilayer_repetitions.
        """
        return np.concatenate(
            [np.tile(sml.susceptibility_array(energy_eV, fERES, gamma2), (rep, 1)) for sml, rep in zip(self._sub_multilayers, self._repetitions)]
            + [np.full((1, energy_eV.shape[0]), self.substrate_susceptibility)]
        )


################################################################
# Reflectivity
################################################################

def create_u_vector(energy_eV, chi, theta):
    """ Calculates the wavevector for each layer from the susceptibility chi
    and incident angle theta
    """
    sin_theta = np.sin(np.deg2rad(theta))
    
    wl_nm = 1e9 * const.h * const.c / energy_eV / const.eV
    k0 = 2 * np.pi / wl_nm

    # add a 0 infront for the region outside the sample
    chi = np.insert(chi, 0, 0, axis=0)
    
    u = - k0 * np.sqrt(sin_theta**2 + chi)
        
    return u

def parratt_formalism(u_vector, t_ML, sigma_ML):
    """ Computes the multilayer reflectivity using the parratt formalism
    Arguments:
        u_vector: (np.ndarray, ndim=2) wavevector for the sample, 
            second dimension for energy
        t_ML: (np.ndarray, ndim=1) thicknesses of the sample layers
        sigma_ML: (np.ndarray, ndim=1) roughnesses of the sample layers
    """

    # expand for compatibility with the second energy dimension
    t_ML = np.expand_dims(t_ML, axis=1)
    sigma_ML = np.expand_dims(sigma_ML, axis=1)

    # Fresnel coefficients
    rj = (( u_vector[:-1] - u_vector[1:] ) / ( u_vector[:-1] + u_vector[1:] )) \
        * np.exp( -2 * sigma_ML**2 * u_vector[:-1] * u_vector[1:] )

    # start for the Parratt formalism
    R = np.zeros(u_vector.shape[1], dtype=np.complex128)

    # Parratt formalism, works from back to front
    for i in reversed(range(u_vector.shape[0]-1)):
        R = np.exp( -2 * 1j * u_vector[i] * t_ML[i] ) \
            * (( rj[i] + R ) / ( 1 + rj[i] * R ))
    
    return R

def reflectivity_parratt(
        energy_eV, 
        multilayer_model: MultilayerModel, 
        theta, 
        fERES = 0, 
        gamma2 = 0
    ):
    """ Computes the multilayer reflectivity from the given parameters
    Arguments:
        energy_eV: energies to be computed over, float or 1D array of floats
        multilayer_model: model of the used multilayer
        theta: X-ray incident angle
        fERES: ERES parameter, 0 for no ERES
        gamma2: ERES parameter, 0 for no ERES
    """
    energy_eV = np.array(energy_eV, ndmin=1)
    u_vector = create_u_vector(
        energy_eV, 
        multilayer_model.susceptibility_array(energy_eV, fERES, gamma2), 
        theta
    )
    R = parratt_formalism(
        u_vector, 
        multilayer_model.thickness_array(), 
        multilayer_model.roughness_array()
    )
    return R

################################################################
# Main programs
################################################################

def load_sample(sample_name: str) -> SampleData:
    return SampleData(f"fit_parameters/{sample_name}_parameters.tab")

def create_ml_model(sample_data: SampleData) -> MultilayerModel:
    top_ml = SubMultilayer("top", ["B4C", "Cu", "SiC"])
    top_ml.init_values(sample_data)
    body_ml = SubMultilayer("body", ["B4C", "Cu", "SiC"])
    body_ml.init_values(sample_data)
    bottom_ml = SubMultilayer("bottom", ["B4C", "Cu", "SiC"])
    bottom_ml.init_values(sample_data)

    mlm = MultilayerModel([top_ml, body_ml, bottom_ml], sample_data)
    mlm.set_multilayer_repetitions("body", 15)

    return mlm

if __name__ == "__main__":
    MaterialManager.add_material("B4C", {"B": 4, "C": 1})
    MaterialManager.add_material("Cu", {"Cu": 1})
    MaterialManager.add_material("SiC", {"Si": 1, "C": 1})

    import matplotlib.pyplot as plt

    # theta - 2theta angle
    theta_exp = 68.54/2


    # replace with file name of fitted sample from GenX 3
    sd = load_sample("M-220706E1_Z_2")
    mlm = create_ml_model(sd)
    mlm.set_multilayer_repetitions("body", 16)

    e_array = np.linspace(900, 1000, num=300)
    refl = np.abs(reflectivity_parratt(e_array, mlm, theta_exp))**2
    plt.plot(e_array, refl)

    plt.xlabel("Energy [eV]")
    plt.show()
