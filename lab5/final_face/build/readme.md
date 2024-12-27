## 
facerec_eigenfaces.cpp 上面define有一個 PC 可以控制現在是
要在pc上面run還是在板子上面run

## face detection
### background
need haarcascades



## face recongnition
### background
the model need image dataset and at.txt(image label)

### image dataset 
* 在dataset下面建立資料夾,一個資料夾是一個人的照片
* 將 png 放入該資料夾

### build image label to train the model
> if you update dataset/ , you need to rebuild at.txt
* python create_csv.py dataset > at.txt

### record 
存放了 record/faceProjection.txt 存放之前偵測過得點在eigenvectors space上的projection 可以透過這個去偵測現在detect到的臉是誰
record/model.xml : 訓練完的model
### build
#### PC
cmake ..
make facerec_eigenfaces
### run
./facerec_eigenfaces at.txt record


