try:
    import candl
    import candl.data
except ImportError:
    raise RuntimeError('Can not find candl. Try running: pip install candl-like')

from cosmosis.datablock import option_section, names
cosmo = names.cosmological_parameters
import numpy as np
import os

class CandlCosmoSISLikelihood:
    """
    A thin wrapper to use candl likelihoods in CosmoSIS.
    """

    def __init__(self, options):
        """
        Read in options from CosmoSIS ini file and initialise requested candl likelihood.
        """

        # Read requested data set from ini and grab a name under which the logl value will be recorded
        data_set_str = options.get_string(option_section, "data_set", default="SPT3G_20192020_lensing_GMV")
        if data_set_str[:11] == "candl.data.":
            # This looks like a short-cut to a pre-installed data set, e.g. "candl.data.SPT3G_2018_Lens"
            self.data_set_file = eval(data_set_str)
            self.name = data_set_str
        else:
            # Assume this to be a path pointing directly to a data set .yaml file
            self.data_set_file = data_set_str
            self.name = "candl." + self.data_set_file.split("/")[-1][:-5]

        # Read in other options from ini
        # Further options for fancy data model selection to come...
        self.lensing = options.get_bool(option_section, 'lensing', default=True)
        self.clear_internal_priors = options.get_bool(option_section, 'clear_internal_priors', default=True)
        self.feedback = options.get_bool(option_section, 'feedback', default=True)
        self.data_selection = options.get_string(option_section, "data_selection", default=None)

        # Initialise the likelihood
        try:
            if self.lensing:
                self.candl_like = candl.LensLike(
                    self.data_set_file,
                    feedback=self.feedback,
                    data_selection=self.data_selection,
                )
            else:
                self.candl_like = candl.Like(
                    self.data_set_file,
                    feedback=self.feedback,
                    data_selection=self.data_selection,
                )
        except:
            raise Exception("candl: likelihood could not be initialised!")

        # by default clear internal priors and assume these are taken care off by CosmoSIS
        if self.clear_internal_priors:
            self.candl_like.priors = [] 

    def reformat(self,block):
        """
        Converting from CosmoSIS to Candl format
        """

        model_dict = {}
        
        # Load all cosmological parameters and add extra params so that candl understands
        # These should not be needed by candl, but it doesn't hurt in case anyone builds a likelihood that directly depends on cosmological parameters
        names      = [param_name for param_sec, param_name in block.keys() if param_sec == 'cosmological_parameters']
        model_dict = {par: block[('cosmological_parameters',par)] for par in names}
        model_dict['H0']   = model_dict['h0']*100
        model_dict['ns']   = model_dict['n_s']
        model_dict['logA'] = model_dict['log1e10as']
        
        # Load all nuisance parameters
        names      = [param_name for param_sec, param_name in block.keys() if param_sec == 'nuisance_parameters']
        model_dict = {par: block[('nuisance_parameters',par)] for par in names}
        
        # Read in Cls from CosmoSIS and save them in dict.
        # CosmoSIS outputs CAMB Cls in unit of l(l+1)/(2pi) even for pp
        # This matches candl expectations for primary CMB - will have to run this and check for lensing
        ell = block[names.cmb_cl, 'ell']
        f1 = ell * (ell + 1) / (2 * np.pi)
        cl_tt = block[names.cmb_cl, 'tt']
        cl_ee = block[names.cmb_cl, 'ee']
        cl_te = block[names.cmb_cl, 'te']
        cl_bb = block[names.cmb_cl, 'bb']
        cl_pp = block[names.cmb_cl, 'pp']
        cl_kk = cl_pp * (ell*(ell+1.))**2. / 4. # ehhh isn't there a factor of pi here?

        # Figure out ell range of supplied spectra w.r.t. the expectation of the likelihood
        N_ell = self.candl_like.ell_max - self.candl_like.ell_min + 1

        theory_start_ix = (
            np.amax((ell[0], self.candl_like.ell_min))
            - ell[0]
        )
        theory_stop_ix = (
            np.amin((ell[-1], self.candl_like.ell_max))
            + 1
            - ell[0]
        )

        like_start_ix = (
            np.amax((ell[0], self.candl_like.ell_min)) - self.candl_like.ell_min
        )
        like_stop_ix = (
            np.amin((ell[-1], self.candl_like.ell_max))
            + 1
            - self.candl_like.ell_min
        )

        # Slot supplied CMB spectra into an array of zeros of the correct length
        # candl will optionally import JAX which should ensure the two methods below run
        model_dict['Dl'] = {'ell': np.arange(self.candl_like.ell_min, self.candl_like.ell_max + 1)}
        for spec_type, spec in zip(["pp", "kk", "TT", "EE", "BB", "TE"] [cl_pp, cl_kk, cl_tt, cl_ee, cl_bb, cl_te]):
            model_dict['Dl'][spec_type] = jnp.zeros(N_ell)
            model_dict['Dl'][spec_type] = jax_optional_set_element(
                model_dict['Dl'][spec_type],
                np.arange(like_start_ix, like_stop_ix),
                spec[theory_start_ix:theory_stop_ix],
            )
        
        return model_dict
    

    def likelihood(self, block):
        '''Computes loglike'''
        logl  = self.candl_like.log_like(self.reformat(block))
        return float(logl)


def setup(options):
    options = SectionOptions(options)
    return CandlCosmoSISLikelihood(options)

def execute(block, config):
    like    = config.likelihood(block)
    dataset = config.dataset
    block[names.likelihoods, "%s_like"%dataset] = like
    return 0
