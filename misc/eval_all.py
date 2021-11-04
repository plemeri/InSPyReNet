import os

sota = os.listdir('configs/SotA')
sota = [os.path.join('configs', 'SotA', i) for i in sota]

ours = os.listdir('configs')
ours = [os.path.join('configs', i) for i in os.listdir('configs') if os.path.isfile(os.path.join('configs', i))]

total = sota + ours

for i in total:
    command = 'tmux new-window \"source ~/.zshrc && conda activate inspyrenet ' 
    command += '&& python run/Eval.py --config {} '.format(i)
    command += ' --stat'
    command += '\"'
    os.system(command)