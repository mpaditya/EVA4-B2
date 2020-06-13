import matplotlib.pyplot as plt
import numpy as np
import torch

def test_misclassified(model, device, test_loader, nimage = 15):
    model.eval()
    images = []
    preds = []
    actual = []
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            #test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True).view_as(target)  # get the index of the max log-probability
            #correct += pred.eq(target.view_as(pred)).sum().item()
            for a,b,c in zip(data, target, pred):
              if b!=c:
                a = a.cpu().numpy()
                b = b.cpu().numpy()
                c = c.cpu().numpy()
                a = (a/2)+0.5
                images.append(a)
                preds.append(c)
                actual.append(b)
                count += 1
              if count == nimage:
                return images, actual, preds


def plot_images(images,actual, preds, classes, nimage=15):
  fig = plt.figure(figsize=(11,8))
  for i in range(nimage):
    ax = fig.add_subplot(3,5,i+1)
    #ax.imshow(np.rollaxis(images[i],0,3).squeeze())
    ax.imshow(np.transpose(images[i], (1, 2, 0)))
    ax.set_title("Actual: " + str(classes[actual[i]]) +  "\n predicted:  " + str(classes[preds[i]]))
  plt.show()