import json

import tqdm
import matplotlib.pylab as plt
import matplotlib.cm as cm

import numpy as np
from numpy import polyfit
from numpy import polyval
from scipy.interpolate import interp1d
import torch.nn.functional as F
import torch as t 
from hyperopt import fmin, tpe, hp

import hyperparams


class LogReg(t.nn.Module):
    def __init__(self, idx):
        t.nn.Module.__init__(self)
        self.lin = t.nn.Linear(len(idx), 2)
        self.idx = idx

    def forward(self, x):
        return self.lin(x[:, self.idx])


def accuracy(student, x,y):
    '''
    Counts accuracy score
    
    student: student model
    x: features
    y: labels
    
    return: accuracy score
    '''
    student.eval()
    total = 0
    correct = 0
    with t.no_grad():
        out = student(x)
        correct += t.eq(t.argmax(out, 1), y).sum()
        total+=len(x)
    student.train()
    return (correct/total).cpu().detach().numpy()


kl = t.nn.KLDivLoss(reduction='batchmean')
sm = t.nn.Softmax(dim=1)

def distill(out, batch_logits, temp):
    '''
    Counts the distillation term of loss function on a current batch
    
    out: output of the student model
    batch_logits: logits of the teacher model
    temp: temperature value
    
    return: loss value
    '''
    g = F.log_softmax(out/temp)
    f = sm(batch_logits/temp)    
    return kl(g, f)


crit = t.nn.CrossEntropyLoss()

def param_loss(batch,model,h):
    '''
    Counts the full loss value
    
    batch: current batch
    model: torch model
    h: metaparameters
    
    return: loss
    '''
    x,y,batch_logits = batch    
    lambda1,temp = h
    lambda2 = 1.0 - lambda1
    out = model(x)    
    distillation_loss = distill(out, batch_logits, temp)
    student_loss = crit(out, y)                
    loss = lambda1 * distillation_loss + lambda2 * student_loss
    return loss


# def param_loss_old(batch,model,h):
#     x,y,batch_logits = batch
#     lambda1,lambda2,temp = h
#     out = model(x)
#     lambda1 = t.clamp(lambda1, 0.01, 0.99)
#     lambda2 = t.clamp(lambda2, 0.01, 0.99)
#     temp = t.clamp(temp, 0.1, 10.0)
#     distillation_loss = distill(out, batch_logits, temp)
#     student_loss = crit(out, y)
#     loss = lambda1 * distillation_loss + lambda2 * student_loss
#     return loss


def hyperparam_loss(batch, model):
    '''
    Counts classification loss
    
    batch: current batch
    model: torch model
    
    return: loss
    '''
    x,y, te = batch
    out = model(x)
    student_loss = crit(out/te, y)            
    return student_loss


def net_training(epoch_num, x_net_train, y_net_train, x_net_test, y_net_test, order, seed=42, lr0=1.0):
    '''
    Teacher network training on synthetic dataset
    
    epoch_num: number of epochs
    x_net_train: train features
    y_net_train: train labels
    x_net_test: test featuers
    y_net_test: test labels
    order: network parameter
    seed: random seed
    lr0: initial learning rate
    
    return:
    '''
    np.random.seed(seed)
    t.manual_seed(seed)
    
    net = LogReg(order)
    optim = t.optim.SGD(net.parameters(), lr=lr0)
    #scheduler = t.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)
    for e in range(epoch_num):
        net.zero_grad()
        loss = crit(net(x_net_train), y_net_train)
        loss.backward()
        optim.step()

        net.eval()
        if e%1000==0:
            print(accuracy(net, x_net_test, y_net_test))
        net.train()
        #scheduler.step()
    return net


#mode={'distil', 'random'}
def synthetic_base(exp_ver, run_num, epoch_num, start_lambda1, start_temp, filename, order, teacher, x_train, y_train, x_test, y_test, lr0=1.0, mode='distil', seed=42):
    '''
    Basic experiment on synthetic dataset
    
    exp_ver: experiment version
    run_num: number of reruns
    epoch_num: number of epochs
    start_lambda1: initial lambda1 value
    start_temp: initial temperature value
    filename: path to file where logs will be stored
    order: network parameter
    teacher: teacher network
    x_train: train features
    y_train: train labels
    x_test: test featuers
    y_test: test labels
    mode: 'distil', 'random'
    seed: random seed
    lr0: initial learning rate
    
    return:
    '''
    np.random.seed(seed)
    t.manual_seed(seed)
    
    for _ in range(run_num):
        lambda1 = start_lambda1
        temp = start_temp
        internal_results = []
        if mode=='random':
            
            lambda1 = t.nn.Parameter(t.tensor(np.random.uniform(low=0.0, high=1.0)), requires_grad=True)
            #lambda2 = t.nn.Parameter(t.tensor(np.random.uniform(low=0.0, high=1.0)), requires_grad=True)
            temp = t.nn.Parameter(10**t.tensor(np.random.uniform(low=-1.0, high=1.0)), requires_grad=True)

        student = LogReg(order)
        optim = t.optim.SGD(student.parameters(), lr=lr0)
        #scheduler = t.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)
        teacher.eval()
        for e in range(epoch_num):
            student.zero_grad()
            out = student(x_train)
            student_loss = crit(out, y_train)
            distillation_loss = distill(out, teacher(x_train), temp)
            loss = lambda1 * student_loss + (1 - lambda1) * distillation_loss
            loss.backward()
            optim.step()

            student.eval()
            if e%200==0:
               
                    internal_results.append({'epoch': e,
                                             'accuracy':float(accuracy(student, x_test, y_test)),
                                             #'temp':float(h[2]),
                                             'temp':float(temp),
                                             'lambda1':float(lambda1),
                                             })# 'lambda2':float(h[1])})

                    print(internal_results[-1])

                    
              
            
            student.train()
            #scheduler.step()
        with open('../log/synthetic_exp'+exp_ver+'_'+filename+'.jsonl', 'a') as out:
                out.write(json.dumps({'results':internal_results, 'version': exp_ver})+'\n')


# mode = {'opt', 'splines'}
# alg_pars = [(epoch_size1, train_splines_every_epoch1),
#             (epoch_size2, train_splines_every_epoch2),
#             (epoch_size3, train_splines_every_epoch3),
#             ...]
def synthetic_opt(exp_ver, run_num, epoch_num, filename, teacher, x_train, y_train, x_test, y_test, alg_pars, lambdas = None, lr0 = 1e-3, lr = 1.0, clip_grad = 10e-3, mode='opt', seed=42):
    '''
    Experiments on synthetic dataset with optimization 
    
    exp_ver: experiment version
    run_num: number of reruns
    epoch_num: number of epochs
    filename: path to file where logs will be stored
    teacher: teacher network
    x_train: train features
    y_train: train labels
    x_test: test featuers
    y_test: test labels
    alg_pars: an array of tuples of two parameters of the algorithm
    lambdas: initial meatparameter values
    lr0: initial learning rate
    lr: scheduler parameter
    clip_grad: gradient norm
    mode: 'opt', 'splines'
    seed: random seed
    
    return: maximum value of validation accuracy if hyperopt is used, else None
    '''
    np.random.seed(seed)
    t.manual_seed(seed)
    
    hist = []
    
    for epoch_size, train_splines_every_epoch in alg_pars:
        for _ in range(run_num):
            internal_results = []

#             if mode == 'opt' or mode == 'splines':
#                 lambda1 = t.nn.Parameter(t.tensor(np.random.uniform(low=-1, high = 1)), requires_grad=True)
#                 lambda2 = t.nn.Parameter(t.tensor(np.random.uniform(low=-1, high=1)), requires_grad=True)
#                 temp = t.nn.Parameter(t.tensor(np.random.uniform(low=-2, high=0)), requires_grad=True)

#             elif mode == 'random':
#                 lambda1 = t.nn.Parameter(t.tensor(np.random.uniform()), requires_grad=True)
#                 lambda2 = t.nn.Parameter(t.tensor(np.random.uniform()), requires_grad=True)
#                 temp = t.nn.Parameter(t.tensor(10**np.random.uniform(low=-1, high=1)), requires_grad=True)

            lambda1 = t.nn.Parameter(t.tensor(np.random.uniform(low=0.0, high=1.0)), requires_grad=True)
            #lambda2 = t.nn.Parameter(t.tensor(np.random.uniform(low=0.0, high=1.0)), requires_grad=True)
            temp = t.nn.Parameter(t.tensor(10**np.random.uniform(low=-1.0, high=1.0)), requires_grad=True)
            
            if lambdas is not None: # non-random initialization
                lambda1.data *= 0
                #lambda2.data *= 0
                temp.data *= 0
                lambda1.data += lambdas[0]
                #lambda2.data += lambdas[1]
                temp.data += lambdas[1]

            # h = [lambda1, lambda2, temp]
            h = [lambda1, temp]

            student = LogReg([0,1,3])
            #optim = t.optim.Adam(student.parameters())
            optim = t.optim.SGD(student.parameters(), lr=lr0) 
            #scheduler = t.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.5)               
            optim2 = t.optim.SGD(h,  lr=lr)
                
            if mode in ['opt', 'splines']:
                hyper_grad_calc = hyperparams.AdamHyperGradCalculator(student, param_loss, 
                                                                      hyperparam_loss, optim, h)

            teacher.eval()
            te = .1 
            for e in range(epoch_num):
                
                if mode == 'splines':
                    e_ = e//epoch_size
                    if e%epoch_size == 0 and e_ % train_splines_every_epoch == 0:
                        spline_hist = []
                        spline_id  = -1
                    spline_id+=1

                if (mode == 'opt') or (mode == 'splines' and e_ % train_splines_every_epoch == 0):
                    #0.1 #1.0 - (e/epoch_num)
                    optim2.zero_grad()
                    hyper_grad_calc.calc_gradients((x_train,y_train,teacher(x_train)), (x_test, y_test, te))

#                     if mode == 'splines':
                    t.nn.utils.clip_grad_value_(h, clip_grad)
                    for h_ in h:
                        if h_.grad is not None:
                            h_.grad = t.where(t.isnan(h_.grad), t.zeros_like(h_.grad), h_.grad)
#                         grads.append([h_.grad.cpu().detach().clone().numpy() for h_ in h])
#                         spline_hist.append([h_.cpu().detach().clone().numpy() for h_ in h])

                    optim2.step()
                    if lambda1 > 1.0:
                        lambda1.data*=0.0
                        lambda1.data+=1.0
                    #if lambda2 > 1.0:
                    #    lambda2.data*=0.0
                    #    lambda2.data+=1.0
                    if temp > 10.0:
                        temp.data*=0.0
                        temp.data+=10.0                        
                    if lambda1 < 0.0:
                        lambda1.data*=0.0
                    #if lambda2 < 0.0:
                    #    lambda2.data*=0.0
                    if temp < 0.1:
                        temp.data*=0.0
                        temp.data+=.1

                if mode == 'splines':
                    if e_ % train_splines_every_epoch == 0:
                        spline_hist.append([h_.cpu().detach().clone().numpy() for h_ in h])
                    else:
                        spline_out = splines(spline_id)
                        lambda1.data *= 0
                        #lambda2.data *= 0
                        temp.data *= 0
                        lambda1.data += spline_out[0]
                        #lambda2.data += spline_out[1]
                        temp.data += spline_out[1]
                    hist.append([h_.grad.cpu().detach().clone().numpy()  for h_ in h])

                optim.zero_grad()
                out = student(x_train)


                loss = param_loss((x_train,y_train,teacher(x_train)), student,h)
#                 elif mode == 'random':
#                     loss = param_loss_old((x_train,y_train,teacher(x_train)), student,h)

                loss.backward()
                optim.step()
                
                if mode == 'splines':
                    if e_ % train_splines_every_epoch == 0 and e%epoch_size == epoch_size-1:
                        fitted1 = np.polyfit(range(len(spline_hist)), np.array(spline_hist)[:,0], 1)
                        fitted2 = np.polyfit(range(len(spline_hist)), np.array(spline_hist)[:,1], 1)
                        splines = lambda x : np.array([np.polyval(fitted1, x), np.polyval(fitted2, x)])

                if mode in ['opt', 'splines']:
                    student.train()

                if e%200==0:
                    student.eval()

                    if mode == 'splines':
                        if e_ %train_splines_every_epoch == 0:
                            mode_opt = 'hypertrain'
                        else:
                            mode_opt = 'hyperpredict'

                    student.train()

#                     if mode == 'opt' or mode == 'splines':
#                         accuracy_final = float(accuracy(student, x_test, y_test))
#                         lambda1_final = float(F.sigmoid(lambda1).detach().numpy())
#                         lambda2_final = float(F.sigmoid(lambda2).detach().numpy())
#                         if mode == 'opt':
#                             temp_final = float(10*F.sigmoid(temp).detach().numpy())
#                         elif mode == 'splines':
#                             temp_final = float(9.9*F.sigmoid(temp).detach().numpy()+0.1)
#                     elif mode == 'random':
#                         accuracy_final = float(accuracy(student, x_test, y_test))
#                         lambda1_final = float(lambda1.detach().numpy())
#                         lambda2_final = float(lambda2.detach().numpy())
#                         temp_final = float(temp.detach().numpy())
                   
                    internal_results.append({'epoch': e,
                                                 'accuracy':float(accuracy(student, x_test, y_test)),
                                                 #'temp':float(h[2]),
                                                 'temp':float(h[1]),
                                                 'lambda1':float(h[0]),
                                                 'te2':te,
                                                 'test loss':crit(student(x_test)/te, y_test).item(),
                                                 
                                                 })# 'lambda2':float(h[1])})
                    
                    print(internal_results[-1])
                #scheduler.step()

#                 elif mode == 'splines' and (e_ % train_splines_every_epoch == 0 and e%epoch_size == epoch_size-1):
#                     fitted1 = np.polyfit(range(len(spline_hist)), np.array(spline_hist)[:,0], 1)
#                     fitted2 = np.polyfit(range(len(spline_hist)), np.array(spline_hist)[:,1], 1)
#                     fitted3 = np.polyfit(range(len(spline_hist)), np.array(spline_hist)[:,2], 1)
#                     splines = lambda x : np.array([np.polyval(fitted1, x), np.polyval(fitted2, x), np.polyval(fitted3, x)])

            if filename is not None: # outer function optimization
                if mode == 'opt' or mode=='no-opt':# or mode == 'random':
                    path = '../log/synthetic_exp'+exp_ver+'_'+filename+'.jsonl'
                elif mode == 'splines':
                    path = '../log/synthetic_exp'+exp_ver+'_'+filename+'_esize_{}_period_{}.jsonl'.format(epoch_size, train_splines_every_epoch)
                with open(path, 'a') as out:
                    out.write(json.dumps({'results':internal_results, 'version': exp_ver})+'\n')
            else:
                 # inner function for hyperopt optimization
                return ([res['accuracy'] for res in internal_results][-1])


def open_data_json(path):
    '''
    Opens log files
    
    path: path to file
    return: data in numpy dataset
    '''
    with open(path, "r") as read_file:
        data = [json.loads(line) for line in read_file]
    return data


def plot_data_params(data, s, label, color, sign):
    '''
    Plot selected data parameters through iterations
    
    data: array with llog data
    s: metaparameter name str
    label: plot label
    color: color of the plot
    sign: plot marker
    
    return:
    '''
#     e = np.array([data[0][i][0] for i in range(len(data[0]))])
#     par = np.array([subdata[i][s] for i in range(len(data[0])) for subdata in data]).reshape(e.shape[0], -1)
#     plt.plot(e, par.mean(1), '-'+sign, color=color, label=label)
#     plt.fill_between(e, par.mean(1)-par.std(1), par.mean(1)+par.std(1), alpha=0.2, color=color)
    e = np.array([data[0]['results'][i]['epoch'] for i in range(len(data[0]['results']))])
    par = np.array([subdata['results'][i][s] for i in range(len(data[0]['results'])) for subdata in data]).reshape(e.shape[0], -1)
    plt.plot(e, par.mean(1), '-'+sign, color=color, label=label)
    plt.fill_between(e, par.mean(1)-par.std(1), par.mean(1)+par.std(1), alpha=0.2, color=color)
    
    
def synthetic_with_hyperopt(exp_ver, run_num, epoch_num, filename, teacher, x_train, y_train, x_test, y_test, alg_pars, lambdas = None, lr0 = 1e-3, lr = 1.0, clip_grad = 10e-3, mode='opt', seed=42, trial_num=5):
    '''
    Inner hyperopt optimization function
    
    exp_ver: experiment version
    run_num: number of reruns
    epoch_num: number of epochs
    filename: path to file where logs will be stored
    alg_pars: an array of tuples of two parameters of the algorithm
    lambdas: initial meatparameter values
    teacher: teacher network
    x_train: train features
    y_train: train labels
    x_test: test featuers
    y_test: test labels
    trial_num: number of reruns
    lr0: initial lr
    lr: scheduler parameter
    clip_grad: gradient norm
    mode: 'opt', 'splines'
    seed: random seed
    
    return:
    '''
    np.random.seed(42)
    t.manual_seed(42)

    for _ in range(run_num):
        cost_function = lambda lambdas: -synthetic_opt(exp_ver, 1, epoch_num, None, teacher, x_train, y_train, x_test, y_test, alg_pars, lambdas = [lambdas[0], 10**lambdas[1]], lr0 = lr0, lr = lr, clip_grad = 10e-3, mode='no-opt', seed=42) # validation accuracy * (-1) -> min
       
        best_lambdas = fmin(fn=cost_function,                             
        #space= [ hp.uniform('lambda1', 0.0, 1.0), hp.uniform('lambda2', 0.0, 1.0), hp.uniform('temp', 0.1, 10.0)],
        space= [ hp.uniform('lambda1', 0.0, 1.0), hp.uniform('temp', -1.0, 1.0)],  
        algo=tpe.suggest,
        max_evals=trial_num)
        #cifar_with_validation_set(exp_ver, 1, epoch_num, filename, tr_s_epoch, m_e, tr_load, t_load, val_load, validate_every_epoch, lambdas = [best_lambdas['lambda1'], best_lambdas['lambda2'], best_lambdas['temp']],  mode='no-opt')
        synthetic_opt(exp_ver, 1, epoch_num, filename, teacher, x_train, y_train, x_test, y_test, alg_pars, lambdas = [best_lambdas['lambda1'], 10**best_lambdas['temp']], lr0 = lr0, lr = lr, clip_grad = 10e-3, mode='no-opt', seed=42)

        
