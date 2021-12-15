import json
import tqdm
import matplotlib.pylab as plt
import matplotlib.cm as cm
import numpy as np
from numpy import polyfit
from numpy import polyval
from scipy.interpolate import interp1d
import torch as t 
from hyperopt import fmin, tpe, hp
import hyperparams
from fashionmnist_net import *

device = 'cuda' if t.cuda.is_available() else 'cpu'

def accuracy(student, t_load):
    '''
    Counts accuracy score
    
    student: student model
    t_load: torch loader
    
    return: accuracy score
    '''
    student.eval()
    total = 0 
    correct = 0
    with t.no_grad():
        for x,y in t_load:
            x = x.to(device)
            y = y.to(device)
            out = student(x)
            correct += t.eq(t.argmax(out, 1), y).sum()
            total+=len(x)
    student.train()
    return (correct/total).cpu().detach().numpy()


kl = nn.KLDivLoss(reduction='batchmean')
sm = nn.Softmax(dim=1)

def distill(out, batch_logits, temp):
    '''
    Counts the distillation term of loss function on a current batch
    
    out: output of the student model
    batch_logits: logits of the teacher model
    temp: temperature value
    
    return: loss value
    '''
    g = sm(out/temp)
    f = F.log_softmax(batch_logits/temp)    
    return kl(f, g)


crit = nn.CrossEntropyLoss()

# определяем функцию потерь как замкнутую относительно аргументов функцию
# нужно для подсчета градиентов гиперпараметров по двухуровневой оптимизации
def param_loss(batch,model,h):
    '''
    Counts the full loss value
    
    batch: current batch
    model: torch model
    h: metaparameters
    
    return: loss
    '''
    x,y,batch_logits = batch    
    lambda1,lambda2,temp = h
    out = model(x)
    distillation_loss = distill(out, batch_logits, temp)
    student_loss = crit(out, y)                
    loss = lambda1 * distillation_loss + lambda2 * student_loss
    return loss

def param_loss_with_reparametrization(batch,model,h):
    '''
    Counts loss with reparametrization
    
    batch: current batch
    model: torch model
    h: metaparameters
    
    return: loss
    '''
    x,y,batch_logits = batch    
    lambda1,lambda2,temp = h
    out = model(x)
    lambda1 = F.sigmoid(lambda1)
    lambda2 = F.sigmoid(lambda2)
    temp = F.sigmoid(temp) * 9.9+0.1
    distillation_loss = distill(out, batch_logits, temp)
    student_loss = crit(out, y)                
    loss = lambda1 * distillation_loss + lambda2 * student_loss
    return loss


# определяем функцию валидационную функцию потерь как замкнутую относительно аргументов функцию
# нужно для подсчета градиентов гиперпараметров по двухуровневой оптимизации
def hyperparam_loss(batch, model):
    '''
    Counts classification loss
    
    batch: current batch
    model: torch model
    
    return: loss
    '''
    x,y = batch
    out = model(x)
    student_loss = crit(out, y)            
    return student_loss


# mode = {'nodistil', 'distil-1', 'distil-2', 'random'}
def fashionmnist_base(exp_ver, run_num, epoch_num, start_lambda1, start_temp, filename, tr_load, t_load, validate_every_epoch, mode='nodistil'):
    '''
    Basic experiment on fashionmnist dataset
    
    exp_ver: experiment version
    run_num: number of reruns
    epoch_num: number of epochs
    start_lambda1: initial lambda1 value
    start_temp: initial temperature value
    filename: path to file where logs will be stored
    tr_load: train loader
    t_load: test loader
    validate_every_epoch: number of iterations between validation
    class_num: number of classes
    mode: 'nodistil', 'distil', 'random'
    logits: teacher model logits
    seed: random seed
    lr0: initial learning rate
    
    return:
    '''
    np.random.seed(42)
    t.manual_seed(42)
    
    if mode != 'nodistil':
        logits = np.load('../code/logits_cnn.npy')
        
    for _ in range(run_num):
        internal_results = []
        
        if mode == 'distil-1':
            lambda1 = 1
            lambda2 = 0
            temp = 1
            
        elif mode == 'distil-2':
            lambda1 = 0
            lambda2 = 1
            temp = 1
            
        elif mode == 'random':
            lambda1 = t.nn.Parameter(t.tensor(np.random.uniform(low=0.0, high=1.0), device=device), requires_grad=True)
            lambda2 = t.nn.Parameter(t.tensor(np.random.uniform(low=0.0, high=1.0), device=device), requires_grad=True)
            temp = t.nn.Parameter(t.tensor(np.random.uniform(low=0.1, high=10.0), device=device), requires_grad=True)
            h = [lambda1, lambda2, temp]
            
        student = FashionMNIST_Net(10).to(device)
        optim = t.optim.Adam(student.parameters())    
        
        for e in range(epoch_num):
            tq = tqdm.tqdm(tr_load)
            losses = []
            
            for batch_id, (x,y) in enumerate(tq):
                x = x.to(device)
                y = y.to(device)
                
                if mode in ['distil-1', 'distil-2', 'random']:
                    batch_logits = t.Tensor(logits[128*batch_id:128*(batch_id+1)]).to(device)
                
                if mode in ['distil-1', 'distil-2', 'nodistil']:
                    student.zero_grad()           
                    out = student(x)
                    student_loss = crit(out, y)
                
                # remove???
                if mode in ['distil-1', 'distil-2']:
                    distillation_loss = distill(out, batch_logits, temp)
                    loss = lambda1 * student_loss + lambda2 * distillation_loss
                    
                elif mode == 'nodistil':
                    loss = student_loss
                    
                elif mode == 'random':
                    optim.zero_grad()
                    loss = param_loss((x,y,batch_logits), student,h)
                    
                losses.append(loss.cpu().detach().numpy())
                loss.backward()
                optim.step()
                tq.set_description('current loss:{}'.format(np.mean(losses[-10:])))        
                
            if e==0 or (e+1)%validate_every_epoch == 0: # если номер эпохи делится на 5 или эпоха - первая             
                test_loss = []
                student.eval()
                
                for x,y in t_load:
                    x = x.to(device)
                    y = y.to(device)                            
                    test_loss.append(crit(student(x), y).detach().cpu().numpy())                 
                    
                test_loss = float(np.mean(test_loss))
                acc = float(accuracy(student, t_load))
                student.train()
                
                if mode in ['distil-1', 'distil-2', 'nodistil']:
                    internal_results.append({'epoch': e, 'test loss':test_loss, 'accuracy':acc})
                    
                elif mode == 'random':
                    internal_results.append({'epoch': e, 'test loss':test_loss, 'accuracy':acc,
                                     'temp':float((h[2]).cpu().detach().numpy()),
                                     'lambda1':float((h[0]).cpu().detach().numpy()),
                                     'lambda2':float((h[1]).cpu().detach().numpy())})

                print (internal_results[-1])

        with open('../log/fashionmnist_exp'+exp_ver+'_'+filename+'.jsonl', 'a') as out:
            out.write(json.dumps({'results':internal_results, 'version': exp_ver})+'\n')            
            
# mode = {'opt', 'splines', 'no-opt'}
def fashionmnist_with_validation_set(exp_ver, run_num, epoch_num, filename, tr_s_epoch, m_e, tr_load, t_load, val_load, validate_every_epoch, lambdas = None,  lr = 1.0, clip_grad = 10e-3, mode='opt', no_tqdm = False):
    '''
    Experiments on fashionmnist dataset with validation
    
    exp_ver: experiment version
    run_num: number of reruns
    epoch_num: number of epochs
    filename: path to file where logs will be stored
    alg_pars: an array of tuples of two parameters of the algorithm
    tr_load: train loader
    t_load: test loader
    val_load: validation loader
    validate_every_epoch: number of iterations between validation
    class_num: number of classes
    lambdas: initial meatparameter values
    lr0: initial learning rate
    lr: scheduler parameter
    clip_grad: gradient norm
    mode: 'opt', 'splines', 'no-opt'
    no_tqdm: whether to use tqdm
    seed: random seed
    logits: teacher model logits
    
    return: maximum value of validation accuracy if hyperopt is used, else None
    '''
    np.random.seed(42)
    t.manual_seed(42)
    
    hist = []
    logits = np.load('../code/logits_cnn.npy')
    for _ in range(run_num):
        internal_results = []
        
        lambda1 = t.nn.Parameter(t.tensor(np.random.uniform(low=-1, high=1), device=device), requires_grad=True)
        lambda2 = t.nn.Parameter(t.tensor(np.random.uniform(low=-1, high=1), device=device), requires_grad=True)
        temp = t.nn.Parameter(t.tensor(np.random.uniform(low=-2, high=0), device=device), requires_grad=True)
        
        if lambdas is not None: # non-random initialization
            lambda1.data *= 0
            lambda2.data *= 0
            temp.data *= 0
            lambda1.data += lambdas[0]
            lambda2.data += lambdas[1]
            temp.data += lambdas[2]
            
        h = [lambda1, lambda2, temp]

        student = FashionMNIST_Net(10).to(device)
        optim = t.optim.Adam(student.parameters())
        optim2 = t.optim.SGD(h, lr=lr)
        if mode in ['splines', 'opt']:
            hyper_grad_calc = hyperparams.AdamHyperGradCalculator(student, param_loss_with_reparametrization,
                                                                  hyperparam_loss, optim, h)

        for e in range(epoch_num):

            
            tq = tqdm.tqdm(zip(tr_load, val_load))
            
            if no_tqdm:
                tq = zip(tr_load, val_load)
                
            losses = []
            for batch_id, ((x,y), (v_x, v_y)) in enumerate(tq):
                
                if mode == 'splines':
                    mini_e = batch_id // m_e
                    if mini_e % tr_s_epoch == 0 and batch_id % m_e  == 0:
                        spline_hist = []
                        spline_id  = -1
                    spline_id += 1
                    
                x = x.to(device)
                y = y.to(device)

                batch_logits = t.Tensor(logits[128*batch_id:128*(batch_id+1)]).to(device)
                
                if mode == 'opt' or (mode == 'splines' and mini_e % tr_s_epoch == 0):
                    v_x = v_x.to(device)
                    v_y = v_y.to(device)
                    optim2.zero_grad()
                    hyper_grad_calc.calc_gradients((x,y,batch_logits), (v_x, v_y))
                    t.nn.utils.clip_grad_value_(h, clip_grad)
                    for h_ in h:
                        h_.grad = t.where(t.isnan(h_.grad), t.zeros_like(h_.grad), h_.grad)
                    optim2.step()
                    
                if mode == 'splines':
                    if mini_e % tr_s_epoch == 0:
                        spline_hist.append([h_.cpu().detach().clone().numpy()  for h_ in h])
                    else:
                        spline_out = splines(spline_id)
                        lambda1.data *= 0
                        lambda2.data *= 0
                        temp.data *= 0
                        lambda1.data += spline_out[0]
                        lambda2.data += spline_out[1]
                        temp.data += spline_out[2]
                    hist.append([h_.grad.cpu().detach().clone().numpy()  for h_ in h])
                    
                optim.zero_grad()
                if mode in ['opt', 'splines']:
                    loss = param_loss_with_reparametrization((x,y,batch_logits), student,h)
                else:
                    loss = param_loss((x,y,batch_logits), student,h)
                losses.append(loss.cpu().detach().numpy())
                loss.backward()
                optim.step()
                
                if not no_tqdm:
                    tq.set_description('current loss:{}'.format(np.mean(losses[-10:])))
                
                if mode == 'splines':
                    if mini_e % tr_s_epoch == 0 and batch_id%m_e == m_e-1:
                        fitted1 = np.polyfit(range(len(spline_hist)), np.array(spline_hist)[:,0], 1)
                        fitted2 = np.polyfit(range(len(spline_hist)), np.array(spline_hist)[:,1], 1)
                        fitted3 = np.polyfit(range(len(spline_hist)), np.array(spline_hist)[:,2], 1)
                        splines = lambda x : np.array([np.polyval(fitted1, x), np.polyval(fitted2, x), np.polyval(fitted3, x)])

            if e==0 or (e+1)%validate_every_epoch == 0:
                test_loss = []
                student.eval()
                for x,y in t_load:
                    x = x.to(device)
                    y = y.to(device)
                    test_loss.append(crit(student(x), y).detach().cpu().numpy())
                test_loss = float(np.mean(test_loss))
                test_loss2 = []
                for x,y in t_load:
                    x = x.to(device)
                    y = y.to(device)
                    test_loss2.append(crit(student(x), y).detach().cpu().numpy())
                print (float(np.mean(test_loss2)))


                acc = float(accuracy(student, t_load))
                student.train()
                if mode in ['opt', 'splines']:
                    internal_results.append({'epoch': e, 'test loss':test_loss, 'accuracy':acc,
                                         'temp':float(0.1+9.9*F.sigmoid(h[2]).cpu().detach().numpy()),
                                         'lambda1':float(F.sigmoid(h[0]).cpu().detach().numpy()),
                                         'lambda2':float(F.sigmoid(h[1]).cpu().detach().numpy())})
                else:
                    val_acc = float(accuracy(student, val_load))
                    internal_results.append({'epoch': e, 'test loss':test_loss, 'accuracy':acc,
                                         'temp':float(h[2].cpu().detach().numpy()),
                                         'lambda1':float(h[0].cpu().detach().numpy()),
                                         'lambda2':float(h[1].cpu().detach().numpy()), 'val acc':val_acc})
                    
                    
                    
                print (internal_results[-1])

        if filename is not None: # outer function optimization
            with open('../log/fashionmnist_exp'+exp_ver+'_'+filename+'.jsonl', 'a') as out:
                out.write(json.dumps({'results':internal_results, 'version': exp_ver})+'\n')
        else:
            # inner function for hyperopt optimization
            return max([res['val acc'] for res in internal_results])
            

def fashionmnist_with_hyperopt(exp_ver, run_num, epoch_num, filename, tr_s_epoch, m_e, tr_load, t_load, val_load, validate_every_epoch, trial_num):
    '''
    Inner hyperopt optimization function
    
    exp_ver: experiment version
    run_num: number of reruns
    epoch_num: number of epochs
    filename: path to file where logs will be stored
    alg_pars: an array of tuples of two parameters of the algorithm
    tr_load: train loader
    t_load: test loader
    val_load: validation loader
    validate_every_epoch: number of iterations between validation
    trial_num: number of reruns
    lr0: initial lr
    logits: teacher model logits
    
    return:
    '''
    np.random.seed(42)
    t.manual_seed(42)

    for _ in range(run_num):
        lambdas = [0.1, 1.0, 1.0]
        
        cost_function = lambda lambdas: -fashionmnist_with_validation_set(exp_ver, 1, epoch_num, None, tr_s_epoch, m_e, tr_load, t_load, val_load, validate_every_epoch, lambdas = lambdas,  mode='no-opt') # validation accuracy * (-1) -> min
        
        cost_function(lambdas)
        best_lambdas = fmin(fn=cost_function,                             
        space= [ hp.uniform('lambda1', 0.0, 1.0), hp.uniform('lambda2', 0.0, 1.0), hp.uniform('temp', 0.1, 10.0)], 
        algo=tpe.suggest,
        max_evals=trial_num)
        fashionmnist_with_validation_set(exp_ver, 1, epoch_num, filename, tr_s_epoch, m_e, tr_load, t_load, val_load, validate_every_epoch, lambdas = [best_lambdas['lambda1'], best_lambdas['lambda2'], best_lambdas['temp']],  mode='no-opt')

    

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
    e = np.array([data[2]['results'][i]['epoch'] for i in range(len(data[2]['results']))])
    par = np.array([subdata['results'][i][s] for i in range(len(data[0]['results'])) for subdata in data]).reshape(e.shape[0], -1)
    plt.plot(e, par.mean(1), '-'+sign, color=color, label=label)
    plt.fill_between(e, par.mean(1)-par.std(1), par.mean(1)+par.std(1), alpha=0.2, color=color)
