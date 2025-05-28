## from PMI-SYSU

from utils import *

class Dataset:
    def __init__(self, usage: str, device: torch.device,
                 P: Union[int, Tuple[int, int]],
                 D: int, N: int,
                 sigma_x: float, sigma_w: float):
        
        if usage == 'experiment':
            self.is_experiment = True
        elif usage == 'theory':
            self.is_experiment = False
        else:
            raise ValueError("Usage should be either 'experiment' or 'theory'")
        
        self.device = device
        self.D = D
        self.N = N
        self.sigma_x = sigma_x
        self.sigma_w = sigma_w

        if self.is_experiment:
            self.P_train, self.P_test = P
            self.data_train = None
            self.data_test = None
            self.weight_train = None
            self.weight_test = None
            self.input_matrix_train = None
            self.input_matrix_test = None
            self.label_train = None
            self.label_test = None
        else:
            self.P = P 
            self.data = None
            self.weight = None
            self.input_matrix = None
            self.label = None
    

    def _generate_data(self, P):
        return torch.normal(0, self.sigma_x, size=(P, self.D, self.N+1), device=self.device)
    
    def generate_data(self):
        if self.is_experiment:
            self.data_train = self._generate_data(self.P_train)
            self.data_test = self._generate_data(self.P_test)
        else:
            self.data = self._generate_data(self.P)


    def _generate_weight(self, P):
        return torch.normal(0, self.sigma_w, size=(P, 1, self.D), device=self.device)

    def generate_weight(self):
        if self.is_experiment:
            self.weight_train = self._generate_weight(self.P_train)
            self.weight_test = self._generate_weight(self.P_test)
        else:
            self.weight = self._generate_weight(self.P)


    def _generate_input_matrix(self, data, weight):
        label = torch.matmul(weight, data) / np.sqrt(self.D)
        input_matrix = torch.cat([data, label], dim=1)
        label_value = label[:, -1, -1].clone()
        input_matrix[:, -1, -1] = 0

        return input_matrix, label_value
    
    def generate_input_matrix(self):
        if self.is_experiment:
            self.input_matrix_train, self.label_train = self._generate_input_matrix(self.data_train, self.weight_train)
            self.input_matrix_test, self.label_test = self._generate_input_matrix(self.data_test, self.weight_test)
        else:
            self.input_matrix, self.label = self._generate_input_matrix(self.data, self.weight)


    def get_dataset(self):
        self.generate_data()
        self.generate_weight()
        self.generate_input_matrix()


    def get_field(self, λ0: float):
        if self.is_experiment:
            raise ValueError("Field calculation only available in theory mode")
        if self.input_matrix is None:
            self.get_dataset()

        C = torch.bmm(self.input_matrix, self.input_matrix.permute(0, 2, 1))
        
        v1 = C[:, -1, :].unsqueeze(2)
        v2 = self.input_matrix[:, :, -1].unsqueeze(1)
        s_mn = torch.bmm(v1, v2) / (self.N + 1)
        
        s_i = s_mn.reshape(self.P, -1, 1)
        s_j = s_mn.reshape(self.P, 1, -1)
        self.J = -torch.bmm(s_i, s_j).mean(dim=0)
        
        self.h = (self.label.reshape(-1, 1, 1) * s_mn).mean(dim=0).reshape(-1)
        
        diag_J = torch.diagonal(self.J)
        self.λ = λ0 - diag_J
