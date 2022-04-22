##Some-networks

Some fnn or cnn networks trained based on MNIST, FASHION MNIST and CIFAR10

---

###models

As mentioned above.

### history

Recordong of Accuracy and Loss during the training process.

```python
import pickle

with open(file_name, 'rb') as f:
    history = pickle.load(f)
```
Structure of history.txt file
```json
{
  "loss": list,
  "accuracy": list,
  "val_loss": list,
  "val_accuracy": list
}
```
e.g.
```python
import pickle
import matplotlib.pyplot as plt

with open('history/mnist_fnn_3x100_tanh.txt', 'rb') as f:
    history = pickle.load(f)
plt.plot(history['accuracy'])
plt.show()
```
![accuracy](Figure_1.png "accuracy")

---
###model.eavaluate
|    | Network                         | Accuracy            | Loss                |
|----|---------------------------------|---------------------|---------------------|
| 0  | cifar10_fnn_3x100_atanh         | 0.49889999628067017 | 2.015043258666992   |
| 1  | cifar10_fnn_3x100_sigmoid       | 0.503000020980835   | 1.5928239822387695  |
| 2  | cifar10_fnn_3x100_tanh          | 0.48649999499320984 | 2.2977473735809326  |
| 3  | cifar10_fnn_3x200_atanh         | 0.505299985408783   | 3.311720609664917   |
| 4  | cifar10_fnn_3x200_sigmoid       | 0.5067999958992004  | 1.9966052770614624  |
| 5  | cifar10_fnn_3x200_tanh          | 0.5060999989509583  | 3.9296910762786865  |
| 6  | cifar10_fnn_3x400_atanh         | 0.53329998254776    | 3.399535894393921   |
| 7  | cifar10_fnn_3x400_sigmoid       | 0.5098000168800354  | 2.925849676132202   |
| 8  | cifar10_fnn_3x400_tanh          | 0.5393000245094299  | 3.422207832336426   |
| 9  | cifar10_fnn_3x50_atanh          | 0.49939998984336853 | 1.5699503421783447  |
| 10 | cifar10_fnn_3x50_sigmoid        | 0.489300012588501   | 1.4880868196487427  |
| 11 | cifar10_fnn_3x50_tanh           | 0.49399998784065247 | 1.6297681331634521  |
| 12 | cifar10_fnn_3x700_atanh         | 0.5432999730110168  | 3.148085355758667   |
| 13 | cifar10_fnn_3x700_sigmoid       | 0.5009999871253967  | 3.9767422676086426  |
| 14 | cifar10_fnn_3x700_tanh          | 0.5516999959945679  | 3.0057172775268555  |
| 15 | cifar10_fnn_5x100_atanh         | 0.492000013589859   | 2.565965414047241   |
| 16 | cifar10_fnn_5x100_sigmoid       | 0.43630000948905945 | 1.7610392570495605  |
| 17 | cifar10_fnn_5x100_tanh          | 0.49540001153945923 | 2.8426408767700195  |
| 18 | fashion_mnist_fnn_3x100_atanh   | 0.8808000087738037  | 0.6536651253700256  |
| 19 | fashion_mnist_fnn_3x100_sigmoid | 0.8809000253677368  | 0.42206162214279175 |
| 20 | fashion_mnist_fnn_3x100_tanh    | 0.878600001335144   | 0.8087016344070435  |
| 21 | fashion_mnist_fnn_3x200_atanh   | 0.8888000249862671  | 0.7942275404930115  |
| 22 | fashion_mnist_fnn_3x200_sigmoid | 0.8827999830245972  | 0.5283091068267822  |
| 23 | fashion_mnist_fnn_3x200_tanh    | 0.8902999758720398  | 0.8138660788536072  |
| 24 | fashion_mnist_fnn_3x400_atanh   | 0.899399995803833   | 0.7473593950271606  |
| 25 | fashion_mnist_fnn_3x400_sigmoid | 0.8859999775886536  | 0.6362820267677307  |
| 26 | fashion_mnist_fnn_3x400_tanh    | 0.8970000147819519  | 0.764274001121521   |
| 27 | fashion_mnist_fnn_3x50_atanh    | 0.8794000148773193  | 0.48898983001708984 |
| 28 | fashion_mnist_fnn_3x50_sigmoid  | 0.8809000253677368  | 0.3694779574871063  |
| 29 | fashion_mnist_fnn_3x50_tanh     | 0.879800021648407   | 0.5303187370300293  |
| 30 | fashion_mnist_fnn_3x700_atanh   | 0.9028000235557556  | 0.73441481590271    |
| 31 | fashion_mnist_fnn_3x700_sigmoid | 0.8877999782562256  | 0.7082392573356628  |
| 32 | fashion_mnist_fnn_3x700_tanh    | 0.8970999717712402  | 0.7608593702316284  |
| 33 | fashion_mnist_fnn_5x100_atanh   | 0.8812999725341797  | 0.8855847716331482  |
| 34 | fashion_mnist_fnn_5x100_sigmoid | 0.8708999752998352  | 0.49079692363739014 |
| 35 | fashion_mnist_fnn_5x100_tanh    | 0.881600022315979   | 0.922964334487915   |
| 36 | mnist_fnn_3x100_atanh           | 0.9775000214576721  | 0.19197066128253937 |
| 37 | mnist_fnn_3x100_sigmoid         | 0.9764000177383423  | 0.2246132791042328  |
| 38 | mnist_fnn_3x100_tanh            | 0.9789999723434448  | 0.16634893417358398 |
| 39 | mnist_fnn_3x200_atanh           | 0.9789000153541565  | 0.1713382750749588  |
| 40 | mnist_fnn_3x200_sigmoid         | 0.978600025177002   | 0.20671476423740387 |
| 41 | mnist_fnn_3x200_tanh            | 0.9805999994277954  | 0.1728319674730301  |
| 42 | mnist_fnn_3x400_atanh           | 0.9821000099182129  | 0.14046545326709747 |
| 43 | mnist_fnn_3x400_sigmoid         | 0.9796000123023987  | 0.16493669152259827 |
| 44 | mnist_fnn_3x400_tanh            | 0.984000027179718   | 0.13373933732509613 |
| 45 | mnist_fnn_3x50_atanh            | 0.9685999751091003  | 0.2519739866256714  |
| 46 | mnist_fnn_3x50_sigmoid          | 0.96670001745224    | 0.16925545036792755 |
| 47 | mnist_fnn_3x50_tanh             | 0.9718000292778015  | 0.24793413281440735 |
| 48 | mnist_fnn_3x700_atanh           | 0.983299970626831   | 0.1303478628396988  |
| 49 | mnist_fnn_3x700_sigmoid         | 0.9800000190734863  | 0.16102369129657745 |
| 50 | mnist_fnn_3x700_tanh            | 0.9843000173568726  | 0.11947346478700638 |
| 51 | mnist_fnn_5x100_atanh           | 0.9767000079154968  | 0.19681191444396973 |
| 52 | mnist_fnn_5x100_sigmoid         | 0.965399980545044   | 0.2509237825870514  |
| 53 | mnist_fnn_5x100_tanh            | 0.9769999980926514  | 0.18982458114624023 |
