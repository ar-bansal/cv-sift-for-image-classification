# Object Classification Using SIFT + BOVW + SVM  


## Objective  
Using scale-invariant feature transform (SIFT) to classify images of playing cards by suit and card number.  

## Dataset  
- **Size**: 7509 training images, 260 validation images, 260 test images  

## Approach  
1. **Feature Extraction**: SIFT descriptors on the whole image. 
2. **Clustering**: BoVW with k-means clustering. Used a GPU to speed up clustering. 
3. **Classification**: Standard scaler + SVM with RBF kernel.  

## Results  
- Best models  
| SVM Kernel  | SVM C | SVM Degree | Training Accuracy | Validation Accuracy |
|-------------|-------|------------|-------------------|---------------------|
| rbf*        | 1.0   | -          | 81.59             | 72.69               |
| poly        | 1.0   | 3          | 84.63             | 67.31               |

The starred model was selected as the best performing model. For that model, grid search was used with poly and rbf kernels. 

- Training confusion matrix for the best model 
|               | True Clubs | True Diamonds | True Hearts | True Spades |  
|---------------|------------|---------------|-------------|-------------|  
| Pred Clubs    | 1478       | 54            | 99          | 164         |
| Pred Diamonds | 60         | 1542          | 144         | 90          |
| Pred Hearts   | 130        | 123           | 1460        | 138         |
| Pred Spades   | 138        | 87            | 103         | 1414        |


- Validation confusion matrix for the best model
|               | True Clubs | True Diamonds | True Hearts | True Spades |  
|---------------|------------|---------------|-------------|-------------|  
| Pred Clubs    | 56         | 2             | 4           | 13          |
| Pred Diamonds | 0          | 50            | 11          | 5           |
| Pred Hearts   | 3          | 7             | 44          | 8           |
| Pred Spades   | 6          | 6             | 6           | 39          |


## Insights  
- Selecting the vocabulary size as k * 10, where k is the number of classes helps speed up clustering without sacrificing accuracy. 
