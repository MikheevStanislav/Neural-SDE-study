import copy
import json
import math
from numbers import Integral
from types import MappingProxyType
import numpy as np
import os
import pathlib
import sklearn.metrics
import torch
import tqdm
import os
import time 
import models_sde as models
from models_sde.neuralsde import (
    _MIXTURE_DEFAULT_OPTIONS,
    _validate_mixture_options,
)

here = pathlib.Path(__file__).resolve().parent


SDE_MODEL_PRESETS = MappingProxyType({
    'staticsde': (1, 0),
    'naivesde': (1, 18),
    'neurallsde': (2, 16),
    'neurallnsde': (4, 17),
    'neuralgsde': (6, 17),
})
GENERIC_SDE_MODEL = 'diffusionsde'
SDE_MODEL_NAMES = frozenset((*SDE_MODEL_PRESETS, GENERIC_SDE_MODEL))


def _add_weight_regularisation(loss_fn, regularise_parameters, mode='l1', scaling=0.01):
    def new_loss_fn(pred_y, true_y):
        total_loss = loss_fn(pred_y, true_y)
        for parameter in regularise_parameters.parameters():
            if parameter.requires_grad:
                if mode == 'l1':
                    # total_loss = total_loss + scaling * torch.norm(parameter, p='nuc')
                    total_loss = total_loss + scaling * torch.norm(parameter, p=1)
                elif mode == 'l2':
                    total_loss = total_loss + scaling * torch.norm(parameter, p='fro')
                else:
                    pass
        return total_loss
    return new_loss_fn


def _nan_safe_mse(pred_y, true_y):
    """MSE over observed target entries only.

    PhysioNet forecast targets are naturally sparse (unobserved bins are
    NaN). When the target has no NaN (e.g. MuJoCo) this reduces to the exact
    same ``torch.nn.functional.mse_loss`` call as before.
    """
    mask = torch.isfinite(true_y)
    if bool(mask.all()):
        return torch.nn.functional.mse_loss(pred_y, true_y)
    masked_squared_error = (pred_y - true_y)[mask].square()
    return masked_squared_error.sum() / mask.sum().clamp_min(1)


def _nan_safe_huber(pred_y, true_y):
    """Huber loss over observed target entries only (see _nan_safe_mse)."""
    mask = torch.isfinite(true_y)
    if bool(mask.all()):
        return torch.nn.functional.huber_loss(pred_y, true_y)
    return torch.nn.functional.huber_loss(pred_y[mask], true_y[mask])

class _SqueezeEnd(torch.nn.Module):
    def __init__(self, model):
        super(_SqueezeEnd, self).__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        
        return self.model(*args, **kwargs).squeeze(-1)


def _count_parameters(model):
    """Counts the number of parameters in a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad_)


class _AttrDict(dict):
    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, item):
        return self[item]


def resolve_sde_config(
    model_name,
    sde_input_option=None,
    sde_noise_option=None,
    sde_mixture_options=None,
):
    """Resolve a named SDE preset or an explicit diffusion configuration."""
    if model_name in SDE_MODEL_PRESETS:
        if (
            sde_input_option is not None
            or sde_noise_option is not None
            or sde_mixture_options is not None
        ):
            raise ValueError(
                f"Model {model_name!r} is a fixed SDE preset and does not "
                "accept --sde_input_option, --sde_noise_option, or "
                "--sde_mixture_options. Use "
                f"--model {GENERIC_SDE_MODEL} for an explicit configuration."
            )
        input_option, noise_option = SDE_MODEL_PRESETS[model_name]
    elif model_name == GENERIC_SDE_MODEL:
        if sde_input_option is None or sde_noise_option is None:
            raise ValueError(
                f"Model {GENERIC_SDE_MODEL!r} requires both "
                "--sde_input_option and --sde_noise_option."
            )
        if (
            isinstance(sde_input_option, bool)
            or not isinstance(sde_input_option, Integral)
            or sde_input_option not in range(7)
        ):
            raise ValueError(
                "sde_input_option must be an integer from 0 to 6; "
                f"got {sde_input_option!r}."
            )
        if (
            isinstance(sde_noise_option, bool)
            or not isinstance(sde_noise_option, Integral)
            or sde_noise_option not in range(25)
        ):
            raise ValueError(
                "sde_noise_option must be an integer from 0 to 24; "
                f"got {sde_noise_option!r}."
            )
        input_option = int(sde_input_option)
        noise_option = int(sde_noise_option)

        if noise_option == 24:
            mixture_options = _validate_mixture_options(
                sde_mixture_options
                if sde_mixture_options is not None
                else _MIXTURE_DEFAULT_OPTIONS
            )
        elif sde_mixture_options is not None:
            raise ValueError(
                "--sde_mixture_options is only valid with --sde_noise_option 24."
            )
        else:
            mixture_options = None
    else:
        raise ValueError(
            f"Unrecognised SDE model name {model_name!r}. Valid names are "
            f"{sorted(SDE_MODEL_NAMES)}."
        )

    diffusion_spec = models.get_diffusion_spec(noise_option)
    config = _AttrDict(
        model_name=model_name,
        input_option=input_option,
        noise_option=noise_option,
        diffusion_name=diffusion_spec['label'],
        raw_diffusion_formula=diffusion_spec['raw_formula'],
        effective_diffusion_formula=diffusion_spec['effective_formula'],
    )
    if noise_option == 24:
        component_specs = [
            models.get_diffusion_spec(option) for option in mixture_options
        ]
        config.mixture = _AttrDict(
            component_options=list(mixture_options),
            component_labels=[
                spec['label'] for spec in component_specs
            ],
            component_raw_formulas=[
                spec['raw_formula'] for spec in component_specs
            ],
            gate_parameterization=(
                "pi = softmax(alpha); alpha learnable, init zeros (uniform)"
            ),
            scale_parameterization=(
                "s = exp(clamp(log_s, -20, 10)); log_s init 0.0"
            ),
            mixture_eps=1e-12,
            effective_formula=(
                "tanh(sigmoid(theta) * s * "
                "sqrt(sum_i(pi_i * g_i(t,y)^2) + eps))"
            ),
        )
    return config


def _evaluate_metrics_forecasting(model_name, dataloader, model, times, loss_fn, device, kwargs):
    with torch.no_grad():
        total_dataset_size = 0
        total_loss = 0
        
        for batch in dataloader:
            batch = tuple(b.to(device) for b in batch)
            *coeffs, true_y, lengths = batch
            batch_size = true_y.size(0)

            pred_y = model(times, coeffs, lengths, **kwargs)
                
            total_dataset_size += batch_size
            total_loss += loss_fn(pred_y, true_y) * batch_size

        total_loss /= total_dataset_size  
        
        metrics = _AttrDict(dataset_size=total_dataset_size, loss=total_loss.item())
    
        return metrics

class _SuppressAssertions:
    def __init__(self, tqdm_range):
        self.tqdm_range = tqdm_range

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is AssertionError:
            self.tqdm_range.write('Caught AssertionError: ' + str(exc_val))
            return True

def _train_loop_forecasting(model_name, train_dataloader, val_dataloader,test_dataloader, model, times, optimizer, loss_fn, eval_fn, max_epochs,
                           writer, device, kwargs, step_mode) :
                           
    model.train()
    best_model = copy.deepcopy(model)
    best_train_loss = math.inf
    
    best_train_loss_epoch = 0
    best_val_loss = math.inf
    best_val_loss_epoch = 0
    
    history = []
    breaking = False
    # scheduler : Reduce learning rate when a metric has stopped improving.
    if step_mode == 'trainloss':
        print("trainloss")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    elif step_mode=='valloss':
        print("valloss")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    elif step_mode == 'valaccuracy':
        print("valaccuracy")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,mode='max')

    elif step_mode=='valauc':
        print("valauc")
        epoch_per_metric = 1
        plateau_terminate = 50
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5,mode='max')
        
    elif step_mode=='none':
        epoch_per_metric=1 
        plateau_terminate=50
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-5)

    tqdm_range = tqdm.tqdm(range(max_epochs))
    tqdm_range.write('Starting training for model:\n\n' + str(model) + '\n\n')
    for epoch in tqdm_range:
        if breaking:
            break
        
        for batch in train_dataloader:
            batch = tuple(b.to(device) for b in batch)
            if breaking:
                break
            with _SuppressAssertions(tqdm_range):
                
                *train_coeffs, train_y, lengths = batch
                
                pred_y = model(times, train_coeffs, lengths, **kwargs)
                loss = loss_fn(pred_y, train_y)
                
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
    

        if epoch % epoch_per_metric == 0 or epoch == max_epochs - 1:
            model.eval()
            # evaluate train,val,test. 
            train_metrics = _evaluate_metrics_forecasting(model_name, train_dataloader, model, times, eval_fn,  device, kwargs)
            val_metrics = _evaluate_metrics_forecasting(model_name, val_dataloader, model, times, eval_fn,  device, kwargs)
            test_metrics = _evaluate_metrics_forecasting(model_name, test_dataloader, model, times, eval_fn,  device,kwargs)
            
            writer.add_scalar('train/loss', train_metrics.loss, epoch)
            writer.add_scalar('validation/loss', val_metrics.loss, epoch)
            writer.add_scalar('test/loss', test_metrics.loss, epoch)
            
            model.train()
            
            if train_metrics.loss * 1.0001 < best_train_loss:
                best_train_loss = train_metrics.loss
                best_train_loss_epoch = epoch
                        
            if val_metrics.loss * 1.0001 < best_val_loss:
                best_val_loss = val_metrics.loss
                best_val_loss_epoch = epoch
                del best_model
                best_model = copy.deepcopy(model)
         
            tqdm_range.write('Epoch: {} | Train loss: {:.3} | Val loss: {:.3} | Test loss : {:.3}'.format(epoch, train_metrics.loss, val_metrics.loss, test_metrics.loss))
            
            if step_mode == 'trainloss':
                scheduler.step(train_metrics.loss)
            elif step_mode=='valloss':
                scheduler.step(val_metrics.loss)
            elif step_mode == 'valaccuracy':
                scheduler.step(val_metrics.accuracy)
            elif step_mode=='valauc':
                scheduler.step(val_metrics.auroc)
            else:
                # scheduler.step()
                pass

                
            history.append(_AttrDict(epoch=epoch, train_metrics=train_metrics, val_metrics=val_metrics))
            # Early stop
            if epoch > best_train_loss_epoch + plateau_terminate:
                tqdm_range.write('Breaking because of no improvement in training loss for {} epochs.'
                                    ''.format(plateau_terminate))
                breaking = True
            
           

    for parameter, best_parameter in zip(model.parameters(), best_model.parameters()):
        parameter.data = best_parameter.data
    return history,epoch


class _TensorEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (torch.Tensor, np.ndarray)):
            return o.tolist()
        else:
            super(_TensorEncoder, self).default(o)


def _save_results(name, result):
    loc = here / 'results' / name
    loc.mkdir(parents=True, exist_ok=True)
    num = -1
    for filename in os.listdir(loc):
        try:
            num = max(num, int(filename))
        except ValueError:
            pass
    result_to_save = result.copy()
    del result_to_save['train_dataloader']
    del result_to_save['val_dataloader']
    del result_to_save['test_dataloader']
    result_to_save['model'] = str(result_to_save['model'])

    num += 1
    with open(loc / str(num), 'w') as f:
        json.dump(result_to_save, f, cls=_TensorEncoder)

        
def main_forecasting(name, model_name, times, train_dataloader, val_dataloader, test_dataloader, device, make_model, max_epochs,
                     lr, weight_decay, loss, reg, scale, writer, kwargs, step_mode, pos_weight=torch.tensor(1)):
    times = times.to(device)
    if device != 'cpu':
        torch.cuda.reset_max_memory_allocated(device)
        baseline_memory = torch.cuda.memory_allocated(device)
    else:
        baseline_memory = None   

    model, regularise_parameters = make_model()
    # mse loss function (masked over observed target entries if any are NaN)
    # loss_fn = torch.nn.functional.huber_loss
    if loss == 'mse':
        loss_fn = _nan_safe_mse
    if loss == 'huber':
        loss_fn = _nan_safe_huber
    loss_fn = _add_weight_regularisation(loss_fn, regularise_parameters, mode=reg, scaling=scale)
    eval_fn = _nan_safe_mse
    model.to(device)
    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # train function
    history = _train_loop_forecasting(model_name, train_dataloader, val_dataloader,test_dataloader, model, times, optimizer, loss_fn, eval_fn, max_epochs,
                                      writer, device, kwargs, step_mode)

    model.eval()
    train_metrics = _evaluate_metrics_forecasting(model_name, train_dataloader, model, times, eval_fn, device, kwargs)
    val_metrics = _evaluate_metrics_forecasting(model_name, val_dataloader, model, times, eval_fn, device, kwargs)
    test_metrics = _evaluate_metrics_forecasting(model_name, test_dataloader, model, times, eval_fn, device,kwargs)

    
    if device != 'cpu':
        memory_usage = torch.cuda.max_memory_allocated(device) - baseline_memory
        print(f"memory_usage:{memory_usage}")
    else:
        memory_usage = None
    result = _AttrDict(name=name,
                       model_name=model_name,
                       times=times,
                       memory_usage=memory_usage,
                       baseline_memory=baseline_memory,
                       train_dataloader=train_dataloader,
                       val_dataloader=val_dataloader,
                       test_dataloader=test_dataloader,
                       model=model.to('cpu'),
                       loss_setting = [loss, reg, scale],
                       parameters=_count_parameters(model),
                       history=history,
                       train_metrics=train_metrics,
                       val_metrics=val_metrics,
                       test_metrics=test_metrics)
    sde_config = getattr(make_model, 'sde_config', None)
    if sde_config is not None:
        result.sde_config = _AttrDict(sde_config.copy())
                    
    if name is not None:
        _save_results(name, result)
    return result


def make_model(name, input_channels, output_channels, hidden_channels, hidden_hidden_channels,
    ode_hidden_hidden_channels, num_hidden_layers, use_intensity, initial, output_time=0,
    sde_input_option=None, sde_noise_option=None, sde_mixture_options=None,
    sde_initialization_seed=None):
    
    print(name)
    
    if name in SDE_MODEL_NAMES:
        sde_config = resolve_sde_config(
            model_name=name,
            sde_input_option=sde_input_option,
            sde_noise_option=sde_noise_option,
            sde_mixture_options=sde_mixture_options,
        )

        def make_vector_field():
            return models.Diffusion_model(
                input_channels=input_channels,
                hidden_channels=hidden_channels,
                hidden_hidden_channels=hidden_hidden_channels,
                num_hidden_layers=num_hidden_layers,
                input_option=sde_config.input_option,
                noise_option=sde_config.noise_option,
                mixture_options=sde_mixture_options,
            )

        def make_forecasting_model(vector_field):
            return models.NeuralSDE_forecasting(
                func=vector_field,
                input_channels=input_channels,
                output_time=output_time,
                hidden_channels=hidden_channels,
                output_channels=output_channels,
                initial=initial,
            )

        if name == GENERIC_SDE_MODEL and sde_initialization_seed is not None:
            initialization_seed = int(sde_initialization_seed)
            sde_config.initialization = _AttrDict(
                scheme='paired_substreams_v1',
                vector_field_seed=initialization_seed,
                forecasting_head_seed=initialization_seed + 1,
                preserves_runtime_rng_state=True,
            )

            def make_model():
                # Diffusion options instantiate different numbers of parameters.
                # Isolated RNG streams keep the shared drift/readout weights and
                # the subsequent data/Brownian RNG state paired across the sweep.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(initialization_seed)
                    vector_field = make_vector_field()
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(initialization_seed + 1)
                    model = make_forecasting_model(vector_field)
                return model, vector_field
        else:
            sde_config.initialization = _AttrDict(
                scheme='legacy_global_rng',
                preserves_runtime_rng_state=False,
            )

            def make_model():
                vector_field = make_vector_field()
                model = make_forecasting_model(vector_field)
                return model, vector_field

        make_model.sde_config = sde_config
    ##    
    elif name == 'ncde':
        def make_model():
            vector_field = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'ncde_forecasting':
         def make_model():
            vector_field = models.FinalTanh(input_channels=input_channels, hidden_channels=hidden_channels,
                                            hidden_hidden_channels=hidden_hidden_channels,
                                            num_hidden_layers=num_hidden_layers)
            model = models.NeuralCDE_forecasting(func=vector_field, input_channels=input_channels,output_time=output_time, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'gruode':
        def make_model():
            vector_field = models.GRU_ODE(input_channels=input_channels, hidden_channels=hidden_channels)
            model = models.NeuralCDE(func=vector_field, input_channels=input_channels,
                                     hidden_channels=hidden_channels, output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name =='gruode_forecasting':
        def make_model():
            vector_field = models.GRU_ODE(input_channels=input_channels, hidden_channels=hidden_channels)
            
            model = models.NeuralCDE_forecasting(func=vector_field, input_channels=input_channels,output_time=output_time, hidden_channels=hidden_channels,
                                     output_channels=output_channels, initial=initial)
            return model, vector_field
    elif name == 'dt':
        def make_model():
            model = models.GRU_dt(input_channels=input_channels, hidden_channels=hidden_channels,
                                  output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'dt_forecasting':
        def make_model():
            model = models.GRU_dt_forecasting(input_channels=input_channels, hidden_channels=hidden_channels,
                                  output_channels=output_channels, use_intensity=use_intensity, output_time = output_time)
            return model, model
    elif name == 'decay':
        def make_model():
            model = models.GRU_D(input_channels=input_channels, hidden_channels=hidden_channels,
                                 output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'decay_forecasting':
        def make_model():
            model = models.GRU_D_forecasting(input_channels=input_channels, hidden_channels=hidden_channels,
                                 output_channels=output_channels, use_intensity=use_intensity, output_time = output_time)
            return model, model
    elif name == 'odernn':
        def make_model():
            model = models.ODERNN(input_channels=input_channels, hidden_channels=hidden_channels,
                                  hidden_hidden_channels=hidden_hidden_channels, num_hidden_layers=num_hidden_layers,
                                  output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    elif name == 'odernn_forecasting':
        def make_model():
            
            model = models.ODERNN_forecasting(input_channels=input_channels,output_time = output_time, hidden_channels=hidden_channels,
                                  hidden_hidden_channels=hidden_hidden_channels, num_hidden_layers=num_hidden_layers,
                                  output_channels=output_channels, use_intensity=use_intensity)
            return model, model
    else:
        raise ValueError("Unrecognised model name {}. Valid names are 'ncde', 'gruode', 'dt', 'decay' and 'odernn'."
                         "".format(name))
    return make_model
