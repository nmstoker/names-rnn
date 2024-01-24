from model import *
from data import *
import sys
from pathlib import Path

def latest_file(path: Path, pattern: str = "*"):
    files = path.glob(pattern)
    return max(files, key=lambda x: x.stat().st_ctime)

model_folder = Path().resolve()
pt_file = latest_file(model_folder,'char-rnn-classification*.pt')
print(f"Using pytorch model: {pt_file}")

rnn = torch.load(pt_file)

# Just return an output given a line
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    
    return output

def predict(line, n_predictions=3):
    output = evaluate(Variable(lineToTensor(line)))

    # Get top N categories
    topv, topi = output.data.topk(n_predictions, 1, True)
    predictions = []

    for i in range(n_predictions):
        value = topv[0][i]
        category_index = topi[0][i]
        print('(%.2f) %s' % (value, all_categories[category_index]))
        predictions.append([value, all_categories[category_index]])

    return predictions

if __name__ == '__main__':
    predict(sys.argv[1])
