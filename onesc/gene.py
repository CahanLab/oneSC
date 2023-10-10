class gene(object): 

    def __init__(self):

        self.upstream_genes = []
        self.regulation_combo = dict()
        self.norm_factors = None 
        self.max_exp = None 
        
        self.regulation_coeff = None 

        self.sigmoid_parameters = dict()
        self.linear_parameters = dict()

    def add_upstream_genes(self, upstream_genes):
        self.upstream_genes = upstream_genes
    
    def add_norm_factors(self, norm_factors):
        self.norm_factors = norm_factors 
        
    def add_regulation_combo(self, regulation_combo):
        self.regulation_combo = regulation_combo

    def add_regulation_coeff(self, regulation_coeff):
        self.regulation_coeff = regulation_coeff
    
    def add_max_exp(self, max_exp): 
        self.max_exp = max_exp
    
    def add_sigmoid_parameters(self, sigmoid_parameters): 
        self.sigmoid_parameters = sigmoid_parameters
    
    def add_linear_parameters(self, linear_parameters):
        self.linear_parameters = linear_parameters