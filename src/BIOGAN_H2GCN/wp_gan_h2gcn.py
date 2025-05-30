import numpy as np 
import random
import torch
import torch.nn as nn
from model_utils import * 
from losses import *
from model_utils_gnn_coex_2 import *  #currently importing just the coexpression processor
# Set seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
from metrics.correlation_score import gamma_coef, gamma_coefficients
from metrics.precision_recall import get_precision_recall
from utils.utils import * 
import sys    
from sklearn.decomposition import PCA
from visualization.visualize import *
from classification.classifier_new import detection
from classification.classifier_new2 import detection,  tissues_classification
from metrics.compute_evaluation_metrics import compute_evaluation_metrics
import time
import wandb
#wandb.require("core")



class WGAN_GP_h2gnn():

    def __init__(self, input_dims, latent_dims, 
                vocab_sizes, 
                generator_dims, 
                discriminator_dims, stat1, stat2, genes,
                learn_graph=True, coex_graph= True, two_tissues= True, random_graph=False,
                masked=False, masking=0.0, decoder_dim= 8, decoder_l= 3,
                decoder_arch = 'graphconv',
                negative_slope = 0.0, is_bn= False,  numerical_dims= [],
                lr_d = 5e-4, lr_g = 5e-4, optimizer='rms_prop', 
                gp_weight = 10,
                p_aug=0, norm_scale=0.5, train=True,
                n_critic = 5,
                freq_print= 2, freq_compute_test = 10, freq_visualize_test=100, patience=10,
                normalization = 'standardize', log2 = False, rpm = False,
                results_dire = '',thr=0.90,rand_seed=1):
        
        print("coex graph", coex_graph)
        print("threshold:", thr)
        self.input_dims = input_dims
        self.latent_dims = latent_dims
        self.numerical_dims = numerical_dims, 
        self.vocab_sizes = vocab_sizes
        self.generator_dims = generator_dims
        self.discriminator_dims = discriminator_dims
        self.negative_slope = negative_slope
        self.is_bn = is_bn
        self.gp_weight = gp_weight
        self.isTrain  = train
        self.p_aug = p_aug
        self.norm_scale = norm_scale
        self.n_genes = input_dims
        self.n_critic = n_critic
        self.two_tissues = two_tissues
        self.freq_print= freq_print
        self.freq_compute_test = freq_compute_test
        self.freq_visualize_test = freq_visualize_test
        self.results_dire = results_dire
        self.results_dire_fig = os.path.join(self.results_dire, 'figures')
        create_folder(self.results_dire)
        create_folder(self.results_dire_fig)
        print(self.results_dire)
        self.dend = False
        self.lr_d = lr_d
        self.thr=thr
        self.rand_seed=rand_seed
        self.lr_g = lr_g
        self.optimizer = optimizer
        self.patience = patience
        # Enabling GPU
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #self.device = 'cpu'
        print('device:', self.device)

        self.genes = genes
        self.output_dim = input_dims
        self.decoder_dim = decoder_dim
        self.decoder_l = decoder_l
        self.patience = patience
        #print('patience:', self.patience)
        self.learn_graph = learn_graph
        self.coex_graph = coex_graph
        self.random_graph = random_graph
        self.masked = masked
        self.masking = masking
        #self.stat1 = stat1
        #self.stat2 = stat2
        self.decoder_arch = decoder_arch
    
        #self.decoder_arch = 'gcn' <-NO

        #self.device = torch.device('cpu')
        #self.loss_fn = utils.set_loss(self.opt,self.device)
        print('numerical_dims:', len(numerical_dims))
        self.loss_dict = {'d loss': [], 
                          'd real loss': [], 
                          'd fake loss': [], 
                          'g loss': []}
        self.corr_scores = {}
        self.corr_dend_scores = {}
        self.precision_scores = {}
        self.recall_scores = {}
        self.stat1 = torch.tensor(stat1, dtype=torch.float32).to(self.device)
        self.stat2 = torch.tensor(stat2, dtype=torch.float32).to(self.device)
        print('1:', self.stat1)
        print('2:', self.stat2)
        self.normalization = normalization
        self.log2 = log2
        self.rpm = rpm
        print('Normalization:', self.normalization)
        print('Log2:', self.log2)
        print('RPM:', self.rpm)

    def init_train(self):

        # Optimizers
        if self.optimizer.lower() == 'rms_prop':
            self.optimizer_disc = torch.optim.RMSprop(self.disc.parameters(), lr=self.lr_d)
            self.optimizer_gen = torch.optim.RMSprop(self.gen.parameters(), lr=self.lr_g)

        
       
        elif self.optimizer.lower() == 'adam':
            betas = (.0, .90)
            self.optimizer_disc = torch.optim.Adam(self.disc.parameters(), lr=self.lr_d, betas=betas)
            self.optimizer_gen = torch.optim.Adam(self.gen.parameters(), lr=self.lr_g, betas=betas)
      
        
    def build_WGAN_GP_gnn(self):

         # to do: fix bug 
        self.numerical_dims = []
        self.gen, self.disc = WGAN_GP_model_gnn(self.latent_dims,  self.input_dims,
                self.numerical_dims, 
                self.vocab_sizes, 
                self.discriminator_dims,  
                self.output_dim, 
                self.decoder_dim, 
                self.device,
                self.genes,
                self.learn_graph,
                self.coex_graph,
                self.two_tissues,
                self.masked,
                self.masking, 
                self.random_graph,
                self.decoder_l,
                self.decoder_arch,
                self.negative_slope, self.is_bn,self.thr,self.rand_seed)
    
        self.disc = self.disc.to(self.device)
        self.gen = self.gen.to(self.device)
            


    def gradient_penalty(self, real_data, fake_data, cat_vars):
    
        batch_size = real_data.size(0)
        alpha= torch.rand(batch_size, 1,
            requires_grad=True,
            device=real_data.device)
        #interpolation =  real_data + torch.mul(alpha, real_data-fake_data)
        interpolation = torch.mul(alpha, real_data) + torch.mul((1 - alpha), fake_data)
        disc_inter_inputs = interpolation
        disc_inter_outputs = self.disc(disc_inter_inputs,  cat_vars)
        grad_outputs = torch.ones_like(disc_inter_outputs)

        # Compute Gradients
        gradients = torch.autograd.grad(
            outputs=disc_inter_outputs,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,)[0]
        
        # Compute and return Gradient Norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)


    def train_disc(self, x, z, cat_vars):

        self.disc.train()
        batch_size = z.shape[0]
        
        # clear existing gradient
        self.optimizer_disc.zero_grad()

        # weights update discriminator
        for w in self.disc.parameters():
            w.requires_grad = True
        
        # no weights update generator 
        for w in self.gen.parameters():
            w.requires_grad = False
        
              # generator input -> concatenate z + numerical vars
       
        gen_inputs = z
        gen_outputs = self.gen(gen_inputs, cat_vars)
        

        
        
        

  
        # augumentation for stability
        # add noise to both fake and true samples
        if self.p_aug != 0:
            augs = torch.distributions.binomial.Binomial().sample(torch.tensor([batch_size])).to(gen_outputs.device)
            #gen_outputs = gen_outputs + augs[:, None] * torch.normal(0, self.norm_scale, size=(self.n_genes,)).to(gen_outputs.device)
            gen_outputs = gen_outputs + augs[:, None] * torch.normal(0, self.norm_scale, size=(self.n_genes,batch_size)).to(gen_outputs.device)
            x = x + augs[:, None] * torch.normal(0, self.norm_scale, size=(self.n_genes, batch_size)).to(x.device)

        disc_fake = self.disc(gen_outputs, cat_vars)
        disc_true = self.disc(x, cat_vars)

        # compute loss
        disc_loss, disc_real_loss, disc_fake_loss = D_loss(disc_true, disc_fake)
        gp = self.gradient_penalty(x, gen_outputs, cat_vars)
        self.disc_loss = disc_loss + self.gp_weight * gp
        # backprop
        self.disc_loss.requires_grad_(True)
        self.disc_loss.backward()
        # update
        self.optimizer_disc.step()

        # save batch loss
        '''self.d_batch_loss = np.array([disc_loss.cpu().detach().numpy().tolist(),
                             disc_real_loss.cpu().detach().numpy().tolist(), 
                             disc_fake_loss.cpu().detach().numpy().tolist()])'''
        self.d_batch_loss = np.array([disc_loss.item(),
                            disc_real_loss.item(), 
                            disc_fake_loss.item()])
        
    def train_gen(self, z, cat_vars):

          
        self.gen.train()
        batch_size = z.shape[0]
        # clear existing gradient
        self.optimizer_gen.zero_grad()
 

         # no weights update discriminator
        for w in self.disc.parameters():
            w.requires_grad = False
        
        #  weights update discriminator
        for w in self.gen.parameters():
            w.requires_grad = True

        gen_inputs = z
        gen_outputs = self.gen(gen_inputs, cat_vars)

        if self.p_aug != 0:
            augs = torch.distributions.binomial.Binomial().sample(torch.tensor([batch_size])).to(gen_outputs.device)
            #gen_outputs = gen_outputs + augs[:, None] * torch.normal(0, self.norm_scale, size=(self.n_genes,)).to(gen_outputs.device)
            gen_outputs = gen_outputs + augs[:, None] * torch.normal(0, self.norm_scale, size=(self.n_genes,batch_size)).to(gen_outputs.device)
            
        disc_fake = self.disc(gen_outputs, cat_vars)
        

        # compute loss
        self.gen_loss = G_loss(disc_fake)
        # backprop
        self.gen_loss.requires_grad_(True)
        self.gen_loss.backward()
        # update
        self.optimizer_gen.step()

        #self.g_batch_loss =np.array([self.gen_loss.cpu().detach().numpy().tolist()])
        self.g_batch_loss =np.array([self.gen_loss.item()])

    def train(self, x_GE, x_cat):
        
    
        x_real = x_GE.clone().to(torch.float32)
        # Train critic 
        for _ in range(self.n_critic):
            z =  torch.normal(0,1, size=(x_real.shape[0], self.latent_dims), device=self.device)
            self.train_disc(x_real, z, x_cat)
        #Train generator 
        z =  torch.normal(0,1, size=(x_real.shape[0], self.latent_dims), device=self.device)
        
        self.train_gen(z, x_cat)

    def generate_samples_all(self, data):

        all_real  = []
        all_gen = []

        all_tissue = []

        for i, data in enumerate(data):
                
            x_GE = data[0].to(self.device)
            x_cat = data[1].to(self.device)
            tissue_t= data[1].t()
            all_tissue.extend(tissue_t[0].numpy())              
            x_real, x_gen = self.generate_samples(x_GE, x_cat)
            if i==0:
                all_gen =  x_gen.cpu().detach().numpy()
                all_real = x_real.cpu().detach().numpy()
            else: 
                all_gen = np.append(all_gen,  x_gen.cpu().detach().numpy(), axis=0)
                all_real = np.append(all_real, x_real.cpu().detach().numpy(), axis=0)

        
        #all_tissue,_ = remap_labels(torch.tensor(all_tissue))
        

        all_real_x = np.vstack(all_real)
        all_real_gen = np.vstack(all_gen)
        
        #print("all_tissue_label", all_tissue)

        return all_real_x, all_real_gen, all_tissue
    

    def generate_samples(self, x_GE,  x_cat):
        
        with torch.no_grad():
            self.gen.eval()
            x_real = x_GE.clone().to(torch.float32)
            z =  torch.normal(0,1, size=(x_real.shape[0], self.latent_dims), device=self.device)
            gen_inputs = z
            x_gen = self.gen(gen_inputs, x_cat)
       
        return x_real, x_gen



    def test(self, test_data, compute_score=True):

        all_real  = []
        all_gen = []

        all_tissue = []
       
        print('----------Testing----------')
        for i, data in enumerate(test_data):
            
        

            x_GE = data[0].to(self.device)
            x_cat = data[1].to(self.device)

            x_real, x_gen = self.generate_samples(x_GE, x_cat)
            if x_real.size(0) == 1:
               x_gen = x_gen.unsqueeze(0)
               print(x_gen.size())

            if i==0:
                all_gen =  x_gen.cpu().detach().numpy()
                all_real = x_real.cpu().detach().numpy()
            else: 
                all_gen = np.append(all_gen,  x_gen.cpu().detach().numpy(), axis=0)
                all_real = np.append(all_real, x_real.cpu().detach().numpy(), axis=0)
            
            data_t = data[1].t()
            all_tissue.extend(data_t[0].numpy())
        
    

        if compute_score:
            all_real_x = np.vstack(all_real)
            all_real_gen = np.vstack(all_gen)
            print(all_real_x.shape)
            print(all_real_gen.shape)
            
            if self.dend:
                pass
                #gamma_dx_dz, gamma_tx_tz = gamma_coefficients(all_real_x, all_real_gen)
                #print(gamma_dx_dz)
                #print(gamma_tx_tz)
                #self.corr_dend_scores[self.epoch +1] = gamma_tx_tz
            else: 
                print('calculating correlation')
                gamma_dx_dz = gamma_coef(all_real_x, all_real_gen)
               # print(gamma_dx_dz)
                print('corr:',gamma_dx_dz)
            #self.corr_scores[self.epoch +1] = float(gamma_dx_dz) 
            prec_10, recall_10 = get_precision_recall(torch.from_numpy(all_real_x), torch.from_numpy(all_real_gen), nb_nn=[10])
            prec_20, recall_20 = get_precision_recall(torch.from_numpy(all_real_x), torch.from_numpy(all_real_gen), nb_nn=[20])
            prec_50, recall_50 = get_precision_recall(torch.from_numpy(all_real_x), torch.from_numpy(all_real_gen), nb_nn=[50])
            print('precision:', prec_10, prec_20, prec_50)
            print('recall:', recall_10, recall_20, recall_50)
            self.precision_scores[self.epoch + 1] = prec_10
            self.recall_scores[self.epoch + 1] = recall_10
            #wandb.log({"correlation": float(gamma_dx_dz), "precision_10":  prec_10, "recall_10": recall_10, "precision_20":  prec_20, "recall_20": recall_20, "precision_50":  prec_50, "recall_50": recall_50})
     
        return all_real, all_gen, all_tissue

    def set_requires_grad(self, nets, requires_grad=False):
   
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


                    
    def fit(self, train_data, val_data, test_data, epochs, val=True):
        torch.cuda.init()
        self.build_WGAN_GP_gnn()

        if self.isTrain:
            self.init_train()

        early_stopper = EarlyStopper_gan(patience=self.patience, min_delta=0)
        for epoch in range(epochs):
            start = time.time()
            if (epoch + 1 ) % 50 == 0:
                 print('reducing learning rate')
                 for param_group in self.optimizer_disc.param_groups:
                     param_group['lr'] = param_group['lr']*0.50
                     print(f"new lr_d: {param_group['lr']}")
                
                 for param_group in self.optimizer_gen.param_groups:
                     param_group['lr'] = param_group['lr']*0.50
                     print(f"new lr_g: {param_group['lr']}")

            train_loss = 0.0
            print('Epoch: ', epoch)
            self.epoch = epoch

            print('----------Training----------')
            d_loss_all = 0.0
            tissues_training=[]
            for i, data in enumerate(train_data):
                
                x_GE = data[0].to(self.device)
                x_cat = data[1].to(self.device)
                
                self.train(x_GE,x_cat)
                d_loss_all += self.disc_loss.item()
                tissue_t= data[1].t()
                tissues_training.extend(tissue_t[0].numpy()) 

                if i==0:
                    d_batch_loss  = self.d_batch_loss 
                    g_batch_loss = self.g_batch_loss
                    
                else:
                    d_batch_loss = d_batch_loss + (self.d_batch_loss)
                    g_batch_loss = g_batch_loss + self.g_batch_loss
            # print loss
                
                if (i+1) % self.freq_print == 0:
                    
                    print('[Epoch %d/%d] [Batch %d/%d] [D loss : %f] [G loss : %f]'
                            %(epoch+1, epochs,       # [Epoch -]
                            i+1,len(train_data),   # [Batch -]
                            self.disc_loss.item(),       # [D loss -]
                            self.gen_loss.item(),       # [G loss -]
                            #loss_GAN.item(),     # [adv -]
                            #loss_cycle.item(),   # [cycle -]
                            ))
                    
            d_batch_loss = d_batch_loss/len(train_data)
            self.loss_dict['d loss'].append(d_batch_loss[0])
            self.loss_dict['d real loss'].append(d_batch_loss[1])
            self.loss_dict['d fake loss'].append(d_batch_loss[2])
            self.loss_dict['g loss'].append(g_batch_loss[0])

            print('Averge D Loss:', d_loss_all/len(train_data))
            end = time.time()
            print('Epoch time:', end-start)
            #wandb.log({"loss d": d_batch_loss[0], "d real loss": d_batch_loss[1],
            #            "d fake loss": d_batch_loss[2],
            #            "loss g": g_batch_loss[0]})
  
            if val:       
                
                if (epoch+1) % self.freq_compute_test == 0:

                    data_real, data_gen,all_tissues_training = self.generate_samples_all(train_data)
                    all_real, all_gen, all_tissue =  self.test(val_data)
                    print('computing the tissues classification.....')
                    #print(f'there are {list(set(all_tissue))} different tissues in the validation set')
                    results = tissues_classification(data_real, all_tissues_training, all_gen ,all_tissue)
                    print('Computing utility (TSTR).......')
                    results = tissues_classification(all_gen, all_tissue ,data_real, all_tissues_training)
                    print('Computing utility (TRTR).......')
                    results = tissues_classification(data_real, all_tissues_training, all_real ,all_tissue)
                    
                    data_real_df = pd.DataFrame(all_real)
                    data_real_df['Tissue'] = all_tissue
                            
                    all_gen_df = pd.DataFrame(all_gen)
                    all_gen_df['Tissue'] = all_tissue
                            
                    #data_real_df.to_csv(os.path.join(f'/home/mongardi/gtex/exps_gtex/tissue_comp', f'data_real_wpgan_{epoch+1}.csv'),index= False)
                    #all_gen_df.to_csv(os.path.join(f'/home/mongardi/gtex/exps_gtex/tissue_comp', f'test_gen_wpgan_{epoch+1}.csv'), index=False)
                    metrics = compute_evaluation_metrics(data_real, data_gen, all_real, all_gen)
                    print(metrics)
                    
                    if (epoch+1) % self.freq_visualize_test == 0:
                        save_checkpoint_gan(self.results_dire, self.gen, self.disc, epoch+1)

                    all_results = {}
                    all_detection = {}
                    all_utility_TSTR ={}
                    all_utility_TRTR ={}
                    all_utility_TRTS ={}
                    
                    if (epoch+1) == epochs:
                        print('plot umap....')
                        plot_umaps(all_real, all_gen, self.results_dire_fig, epoch+1, all_tissue,  n_neighbors=300)
            
                    #metrics = compute_evaluation_metrics(data_real, data_gen, all_real, all_gen)
                        
                        n_runs = 10
                        for run in range(n_runs):
                            print('run:', run)
                        
                            data_real_, data_gen_, all_tissues_training= self.generate_samples_all(train_data)
                            val_real, val_gen, all_tissue=  self.generate_samples_all(val_data)
                            data_real = np.concatenate((data_real_, val_real), axis=0)
                            data_gen = np.concatenate((data_gen_, val_gen), axis=0)  
                            all_real, all_gen, all_tissue  =  self.test(test_data)
                            print("data real", data_real)
                            data_real_renorm,data_gen_renorm,all_real_renorm, all_gen_renorm= reverse_normalization(data_real, data_gen, 
                                                all_real, all_gen, self.stat1,
                                                self.stat2, self.normalization, self.log2,self.rpm)
            
                            
                            results_dire_run = os.path.join(self.results_dire, f"test_{run}_epoch_{epoch+1}")
                            create_folder(results_dire_run)
                            save_numpy(results_dire_run + '/data_real.npy', data_real_renorm)
                            save_numpy(results_dire_run + '/data_gen.npy', data_gen_renorm)
                            save_numpy(results_dire_run + '/test_real.npy', all_real_renorm)
                            save_numpy(results_dire_run + '/test_gen.npy', all_gen_renorm)
                            
                            #data_real_df = pd.DataFrame(all_real)
                            #data_real_df['Tissue'] = all_tissue
                            
                            #all_gen_df = pd.DataFrame(all_gen)
                            #all_gen_df['Tissue'] = all_tissue
                            
                            #data_real_df.to_csv(os.path.join('/home/mongardi/gtex/exps_gtex/tissue_comp', f'data_real_gnn_{epoch+1}.csv'), index=False)
                            #all_gen_df.to_csv(os.path.join('/home/mongardi/gtex/exps_gtex/tissue_comp', f'data_gen_gnn_{epoch+1}.csv'), index=False)
                        
                            dict_data_real = {'data_train': data_real, 'data_test': all_real}
                            dict_data_real = {'data_train': data_gen, 'data_test': all_gen}
                            
                            
                            print("computing last evaluation with renormalized data........")

                            all_results[str(run)+ '_corr']= gamma_coef(all_real, all_gen)
                            metrics = compute_evaluation_metrics(data_real, data_gen, all_real, all_gen)
                            all_results[str(run)] = metrics
                            print(metrics)
                            #wandb.log({"accuracy detection": metrics['Logistic results'][1], "f1 detection":  metrics['Logistic results'][0]})
                            #wandb.log({"accuracy detection pca": metrics['Logistic PCA results'][1], "f1 detection pca":  metrics['Logistic PCA results'][0]})
                            print('-------------------------------------------------------------------------------------------')
                            print(f"Detection complete feature space with {data_real.shape[1]} features")
                            results_detection = detection(data_real, data_gen, all_real, all_gen)
                            all_detection[str(run)] = results_detection
                            acc = []
                            f1 = []
                            auc = []
                            for model_name in results_detection:
                        
                                acc.append(results_detection[model_name]['accuracy'][0])
                                f1.append(results_detection[model_name]['f1'][0])
                                auc.append(results_detection[model_name]['auc'][0])
                        
                            print(f"Model: {model_name}, Accuracy: {results_detection[model_name]['accuracy']}, F1: {results_detection[model_name]['f1']}, 'AUC': {results_detection[model_name]['auc']}")
                            print('-------------------------------------------------------------------------------------------')
                            
                            n_components = 100
                            pca = PCA(n_components=n_components)
                            pca_train_data = pca.fit_transform(data_real)
                            pca_gen_data = pca.transform(data_gen)
                            pca_data_real_test = pca.transform(all_real)
                            pca_data_fake_test = pca.transform(all_gen)
                            print(f"Detection PCA space with {pca_data_real_test.shape[1]} PCs")
                            results_detection = detection(pca_train_data, pca_gen_data, 
                                             pca_data_real_test, pca_data_fake_test)

                            all_detection[str(run) + '_PCA'] = results_detection
                            acc = []
                            f1 = []
                            auc = []
                            for model_name in results_detection:
                        
                                acc.append(results_detection[model_name]['accuracy'][0])
                                f1.append(results_detection[model_name]['f1'][0])
                                auc.append(results_detection[model_name]['auc'][0])
                        
                            print(f"Model: {model_name}, Accuracy: {results_detection[model_name]['accuracy']}, F1: {results_detection[model_name]['f1']}, 'AUC': {results_detection[model_name]['auc']}")
                            
                            print(f"TRTS complete feature space with {data_real.shape[1]} features")
                            results_utility_TRTS = tissues_classification(data_real_, all_tissues_training ,data_gen_, all_tissues_training)
                            all_utility_TRTS[str(run)] = results_utility_TRTS
                            acc = []
                            balanced_acc = []
                            f1 = []
                            f1_weighted = []
                            for model_name in results_utility_TRTS:
                        
                                acc.append(results_utility_TRTS[model_name]['accuracy'][0])
                                balanced_acc.append(results_utility_TRTS[model_name]['balanced accuracy'][0])
                                f1.append(results_utility_TRTS[model_name]['f1'][0])
                                f1_weighted.append(results_utility_TRTS[model_name]['f1_weighted'][0])
                        
                            
                            print('-------------------------------------------------------------------------------------------')                                                
                            
                            
                            
                            
                            print(f"TSTR complete feature space with {data_real.shape[1]} features")
                            results_utility_TSTR = tissues_classification(data_gen_, all_tissues_training ,all_real, all_tissue)
                            all_utility_TSTR[str(run)] = results_utility_TSTR
                            acc = []
                            balanced_acc = []
                            f1 = []
                            f1_weighted = []
                            for model_name in results_utility_TSTR:
                        
                                acc.append(results_utility_TSTR[model_name]['accuracy'][0])
                                balanced_acc.append(results_utility_TSTR[model_name]['balanced accuracy'][0])
                                f1.append(results_utility_TSTR[model_name]['f1'][0])
                                f1_weighted.append(results_utility_TSTR[model_name]['f1_weighted'][0])
                        
                            
                            print('-------------------------------------------------------------------------------------------')   


                            print(f"TRTR complete feature space with {data_real.shape[1]} features")
                            results_utility_TRTR = tissues_classification(data_real_, all_tissues_training ,all_real, all_tissue)
                            all_utility_TRTR[str(run)] = results_utility_TRTR
                            acc = []
                            balanced_acc = []
                            f1 = []
                            f1_weighted = []
                            for model_name in results_utility_TRTR:
                        
                                acc.append(results_utility_TRTR[model_name]['accuracy'][0])
                                balanced_acc.append(results_utility_TRTR[model_name]['balanced accuracy'][0])
                                f1.append(results_utility_TRTR[model_name]['f1'][0])
                                f1_weighted.append(results_utility_TRTR[model_name]['f1_weighted'][0])
                        
                            
                            print('-------------------------------------------------------------------------------------------')   

                            print('Training completed!')    

                                

        
        save_dictionary(self.corr_scores, os.path.join(self.results_dire, 'correlation_scores'))
        save_dictionary(self.precision_scores, os.path.join(self.results_dire, ' precision_scores'))
        save_dictionary(self.recall_scores, os.path.join(self.results_dire, ' recall_scores'))
        print(self.corr_scores)
        #self.print_best_epoch(self.corr_scores)
        #self.print_best_epoch(self.precision_scores, name='precision')
        #self.print_best_epoch(self.recall_scores, name='recall')
        plot_curves(self.loss_dict, self.results_dire_fig)
        
        if self.dend:
            save_dictionary(self.corr_dend_scores, os.path.join(self.results_dire, 'correlation_dendro_scores'))
            self.print_best_epoch(self.corr_dend_scores, name ='correlation dend')
        
        return self.loss_dict
    
    def print_best_epoch(self, d, name='correlation'):
        
        idx_max = max(d, key=d.get)
        print('Best epoch ' + name + ':', idx_max, 'score:', d[idx_max])
    
      # Evaluation metrics
    def score_fn(x_test, cat_covs_test, num_covs_test):

        def _score(gen):
            x_gen = predict(cc=cat_covs_test,
                            nc=num_covs_test,
                            gen=gen)

            gamma_dx_dz = gamma_coefficients(x_test, x_gen)
            return gamma_dx_dz
            # score = (x_test - x_gen) ** 2
            # return -np.mean(score)

        return _score
   