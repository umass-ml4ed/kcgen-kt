import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from omegaconf import OmegaConf
from datetime import datetime
import hydra
import transformers
import wandb

from data_loader import *
from model import *
from trainer import *
from utils import *
from eval import *

@hydra.main(version_base=None, config_path=".", config_name="configs_kc")
def main(configs):
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.empty_cache()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(now)

    # Make reproducible
    set_random_seed(configs.seed)

    # Single GPU setup
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if configs.use_cuda: 
        if torch.cuda.is_available():
            device = torch.device('cuda')
        assert device.type == 'cuda', 'No GPU found'


    # # Use wandb to track experiment
    if configs.log_wandb:
        wandb.login(key=configs.wandb_key, verify=True)
        wandb.init(project=configs.wandb_project)
        print('Run id:', wandb.run.id)
        wandb.config.update(OmegaConf.to_container(configs, resolve=True))


    if configs.baseline:
        kc_problem_dict, kc_no_dict = extract_baseline_kc('data/prompt_concept.csv')

    else:
        kc_problem_dict, kc_no_dict = get_problem_kc(configs.kc_path)  


    ## load the init dataset
    train_stu, valid_stu, test_stu, df, students = read_data('data/dataset_time.pkl', kc_problem_dict, configs)
   

    ## load model 
    if not configs.transition:
        lstm, model, tokenizer = create_model(configs, device, len(kc_no_dict))
        trans_linear = None
    else:
        lstm, model, tokenizer = create_model(configs, device, configs.transition_dim)
        trans_linear = create_knowledge_linear(device, len(kc_no_dict), configs.transition_dim)


    predictor = None
    if configs.multitask:
        if configs.binary_loss_fn == 'BCE':
            predictor = create_multitask_predictor(device, configs.predictor_multilayer)

        else:
            predictor = create_binary_predictor(device)
    
    collate_fn = CollateForKC(tokenizer, configs, device, kc_no_dict)
    
    if configs.testing:
        train_stu = train_stu[:3]
        valid_stu = valid_stu[:3]
        test_stu = test_stu[:3]
        configs.epochs = 1


    train_loader = make_dataloader(train_stu, df, collate_fn, configs)
    valid_loader = make_dataloader(valid_stu, df, collate_fn, configs)
    test_loader = make_dataloader(test_stu, df, collate_fn, configs)


    # optimizater and loss function
    optimizers_generator = []
    optimizer_lm = optim.AdamW(model.parameters(), lr=configs.lr)
    optimizers_generator.append(optimizer_lm)

    optimizers_lstm = []
    optimizer_lstm = optim.RMSprop(lstm.parameters(), lr=configs.lstm_lr, momentum=0.9)
    optimizers_lstm.append(optimizer_lstm)

    optimizers_predictor = None
    binary_loss_fn = None

    optimizers_transition = None
    if configs.transition:
        optimizers_transition = []
        optimizer_trans = optim.AdamW(trans_linear.parameters(), lr=configs.trans_linear_lr)
        optimizers_transition.append(optimizer_trans)

    # KC loss function
    kc_loss_fn = nn.BCELoss(reduction='none')


    if configs.multitask:
        optimizers_predictor = []
        optimizer_predictor = optim.AdamW(predictor.parameters(), lr=configs.pred_linear_lr)
        optimizers_predictor.append(optimizer_predictor)

        if configs.binary_loss_fn == 'BCE':
            binary_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        
        else:
            binary_loss_fn = nn.CrossEntropyLoss(reduction='none')


    # LR scheduler
    num_training_steps = len(train_loader) * configs.epochs
    num_warmup_steps = configs.warmup_ratio * num_training_steps
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer_lm, num_warmup_steps, num_training_steps)


    ## start training
    best_valid_metrics =  {'loss': float('inf')} 
    best_test_metrics =  {'loss': float('inf')} 
    best_metrics_with_valid =  {'loss': float('inf')} 
    train_dl_len = len(train_loader)

    best_test_acc = -float('inf')

    for ep in tqdm(range(configs.epochs), desc="epochs", mininterval=20.0):
        train_logs, test_logs, valid_logs = [], [], []

        # training
        for idx, batch in enumerate(tqdm(train_loader, desc="training", leave=False)):
            train_log = generator_step(idx, batch, model, lstm, tokenizer, optimizers=optimizers_generator, optimizers_lstm=optimizers_lstm,
                   configs=configs, train_dl_len=train_dl_len, train=True, scheduler=scheduler, device=device, group_size=2, multitask=configs.multitask,
                    predictor=predictor, pred_loss_fn=binary_loss_fn, optimizers_multitask=optimizers_predictor, kc_loss_fn=kc_loss_fn, trans_linear=trans_linear, 
                    optimizers_trans=optimizers_transition)
            
            train_logs.append(train_log)

            
            ## save results to wandb
            if configs.log_train_every_itr and configs.log_wandb:
                if (idx+1) % configs.log_train_every_itr == 0:
                    itr_train_logs = aggregate_metrics(train_logs, configs)
                    for key in itr_train_logs:
                        wandb.log({"metrics/train_every_{}_itr/{}".format(configs.log_train_every_itr,key): itr_train_logs[key]})

        ## validation
        for idx, batch in enumerate(tqdm(valid_loader, desc="validation", leave=False)):
            valid_log = generator_step(idx, batch, model, lstm, tokenizer, optimizers=None, optimizers_lstm=None,
                   configs=configs, train_dl_len=train_dl_len, train=False, scheduler=None, device=device, 
                   group_size=2, multitask=configs.multitask, predictor=predictor, pred_loss_fn=binary_loss_fn, 
                   kc_loss_fn=kc_loss_fn, trans_linear=trans_linear, optimizers_trans=optimizers_transition)

            valid_logs.append(valid_log)
        
        ## testing
        for idx, batch in enumerate(tqdm(test_loader, desc="testing", leave=False)):
            test_log = generator_step(idx, batch, model, lstm, tokenizer, optimizers=None, optimizers_lstm=None,
                   configs=configs, train_dl_len=train_dl_len, train=False, scheduler=None, device=device, group_size=2, 
                   multitask=configs.multitask, predictor=predictor, pred_loss_fn=binary_loss_fn, kc_loss_fn=kc_loss_fn,
                   trans_linear=trans_linear, optimizers_trans=optimizers_transition)

            test_logs.append(test_log)
        

        # logging
        train_logs = aggregate_metrics(train_logs, configs)
        valid_logs = aggregate_metrics(valid_logs, configs)
        test_logs  = aggregate_metrics(test_logs, configs )

        ## log the results and save models
        for key in valid_logs:
            ## weighted loss is the total loss among all terms, the loss in kc_identification set up is the prediction loss
            if key == 'loss':
                if( float(valid_logs[key]) < best_valid_metrics[key] ):
                    best_valid_metrics[key] = float(valid_logs[key])
                    for key_best_metric in best_metrics_with_valid:
                        best_metrics_with_valid[key_best_metric] = float(test_logs[key_best_metric])
                    ## Save the model with lowest validation loss
                    print('Saved at Epoch:', ep)
                    print('Best model stats:', test_logs)
                    if configs.save_model:
                        if configs.log_wandb:
                            wandb.log({"best_model_at_epoch": ep, "best_valid_loss": best_valid_metrics[key]})

                    model_dir = os.path.join(configs.model_save_dir, now, 'model')
                    model.save_pretrained(model_dir)

                    torch.save(lstm.state_dict(), os.path.join(configs.model_save_dir, now, 'lstm'))
                    optimizer_lstm_dir = os.path.join(configs.model_save_dir, now, 'optimizer_lstm.pth')
                    torch.save(optimizer_lstm.state_dict(), optimizer_lstm_dir)

                    if configs.transition:
                        torch.save(trans_linear.state_dict(), os.path.join(configs.model_save_dir, now, 'transition'))
                        optimizer_trans_dir = os.path.join(configs.model_save_dir, now, 'optimizer_trans.pth')
                        torch.save(optimizer_trans.state_dict(), optimizer_trans_dir)

                    if configs.multitask:
                        torch.save(predictor.state_dict(), os.path.join(configs.model_save_dir, now, 'predictor'))
                        optimizer_predictor_dir = os.path.join(configs.model_save_dir, now, 'optimizer_predictor.pth')
                        torch.save(optimizer_predictor.state_dict(), optimizer_predictor_dir)
    

                    scheduler_dir = os.path.join(configs.model_save_dir, now, 'scheduler.pth')
                    torch.save(scheduler.state_dict(), scheduler_dir)

                    optimizer_lm_dir = os.path.join(configs.model_save_dir, now, 'optimizer_lm.pth')
                    torch.save(optimizer_lm.state_dict(), optimizer_lm_dir)

        for key in test_logs:
            if key == 'loss':
                if float(test_logs[key])<best_test_metrics[key]:
                    best_test_metrics[key] = float(test_logs[key])
        
        ## save results to wandb:
        if configs.log_wandb:
            saved_stats = {}
            for key in train_logs:
                saved_stats["metrics/train/"+key] = train_logs[key]
            for key in valid_logs:
                saved_stats["metrics/valid/"+key] = valid_logs[key]
            for key in test_logs:
                saved_stats["metrics/test/"+key] = test_logs[key]
            for key in best_test_metrics:
                saved_stats["metrics/test/best_"+key] = best_test_metrics[key]
            for key in best_metrics_with_valid:
                saved_stats["metrics/test/best_"+key+"_with_valid"] = best_metrics_with_valid[key]
            saved_stats["epoch"] = ep

            wandb.log(saved_stats)
    
        if test_logs['acc'] > best_test_acc:
            best_test_acc = test_logs['acc']
    
    print('Best Test Set Accuracy:', best_test_acc)

    # Evaluation process
    collate_fn_eval = CollateForKC(tokenizer, configs, device, kc_no_dict, eval=True)
    test_loader = make_dataloader(test_stu, df, collate_fn_eval, configs)

    print('start eval func:')
    res = evaluate(configs, now, test_loader, tokenizer, device, kc_no_dict)

    
    result = {'codeBLEU': res['codebleu']}
    if configs.multitask:
        result['Acc'] = res['Acc']
        result['AUC']= res['AUC']
        result['F1'] = res['F1']

    print(result)

    if configs.log_wandb:
        wandb.log(result)
        wandb.finish()
    

if __name__ == '__main__':
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    main()