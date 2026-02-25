'''Class structure describing external importations'''
from abc import ABC
from numpy import exp, zeros
from scipy.interpolate import interp1d
from scipy.sparse import csc_matrix as sparse


class ImportModel(ABC):
    '''Abstract class for importation models'''
    def __init__(self,
                no_inf_compartments,
                no_age_classes):
        self.no_inf_compartments = no_inf_compartments
        self.no_age_classes = no_age_classes
        self.no_entries = no_inf_compartments * no_age_classes

        self.default_call_property = "cases" # Choose whether to use the cases or matrix property when called from RateEquations

    def cases(self, t):     # Cases is a list of import functions
        pass


class NoImportModel(ImportModel):
    def cases(self, t):
        return zeros(self.no_age_classes,)

    @classmethod
    def make_from_spec(cls, spec, det):
        return cls()


class FixedImportModel(ImportModel):
    def __init__(
            self,
            no_inf_compartments,
            no_age_classes,
            rhs,
            x0):
        '''import_arrays should be a list of arrays. The number of arrays is
        no_inf_compartments and each has length no_age_classes. The jth element of
        the ith array is the rate at which individuals in age class j are infected
        by external cases in infectious compartment i.'''
        super().__init__(no_inf_compartments, no_age_classes)
        self.x0 = x0
        base_inf_rates = rhs.states_sus_only * (
            rhs.household_population.model_input.k_ext).dot(x0)
        total_size = len(rhs.household_population.which_composition)
        matrix_shape = (total_size, total_size)
        inf_event_row = rhs.household_population.inf_event_row
        inf_event_col = rhs.household_population.inf_event_col
        inf_event_class = rhs.household_population.inf_event_class
        self.base_matrix = sparse(matrix_shape, )
        self.base_matrix += sparse((base_inf_rates[inf_event_row, inf_event_class],
                                    (inf_event_row,
                                     inf_event_col)),
                                   shape=matrix_shape) - \
                            sparse((base_inf_rates[inf_event_row, inf_event_class],
                                    (inf_event_row,
                                     inf_event_row)),
                                   shape=matrix_shape)

        self.default_call_property = "matrix" #In this case using matrix is more efficient

    def cases(self, t):
        return self.x0

    def matrix(self, t):
        '''This calculates an import rate matrix rather than
        the external prevalence, which might improve performance.'''
        return self.base_matrix

class StepImportModel(ImportModel):
    '''This class provides an inefficient way of implementing a step function for external imports. For any reasonably
    long increment it will be more efficient to use the FixedImportModel class, and perform successive solves between
    the increments of the step function.'''
    def __init__(
            self,
            no_inf_compartments,
            no_age_classes,
            time,
            external_prevalance):       # External prevalence is now a age classes by inf compartments array
        super().__init__(no_inf_compartments, no_age_classes)
        self.prevalence_interpolant = []
        for i in range(self.no_entries):
            self.prevalence_interpolant.append(interp1d(
                time, external_prevalance[i,:],
                kind='nearest',
                bounds_error=False,
                fill_value='extrapolate',
                assume_sorted=True))

        self.default_call_property = "cases" #In this case there is no matrix property

    def cases(self, t):
        imports = zeros(self.no_entries,)
        for i in range(self.no_entries):
            imports[i] = self.prevalence_interpolant[i](t)
        return imports


class ExponentialImportModel(ImportModel):
    '''Couples the model to an exponentially growing supply of infection.
    To use this model, a RateEquations object with some other import term
    needs to be constructed and provided as an input argument.'''
    def __init__(self,
                 no_inf_compartments,
                 no_age_classes,
                 rhs,
                 growth_rate,
                 x0):
        super().__init__(no_inf_compartments, no_age_classes)
        self.growth_rate = growth_rate
        self.x0 = x0
        base_inf_rates = rhs.states_sus_only * (
                    rhs.household_population.model_input.k_ext).dot(x0)
        total_size = len(rhs.household_population.which_composition)
        matrix_shape = (total_size, total_size)
        inf_event_row = rhs.household_population.inf_event_row
        inf_event_col = rhs.household_population.inf_event_col
        inf_event_class = rhs.household_population.inf_event_class
        self.base_matrix = sparse(matrix_shape, )
        self.base_matrix += sparse((base_inf_rates[inf_event_row, inf_event_class],
                                   (inf_event_row,
                                    inf_event_col)),
                                  shape=matrix_shape) - \
                            sparse((base_inf_rates[inf_event_row, inf_event_class],
                                    (inf_event_row,
                                     inf_event_row)),
                                   shape=matrix_shape)

        self.default_call_property = "matrix" #In this case using matrix is more efficient

    def cases(self, t):
        return exp( self.growth_rate * t) * self.x0

    def matrix(self, t):
        '''This calculates an import rate matrix rather than
        the external prevalence, which might improve performance.'''
        return exp( self.growth_rate * t) * self.base_matrix