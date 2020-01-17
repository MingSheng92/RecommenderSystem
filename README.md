# Work in progress


## Recommender System

Recommender system, as one of the subclasses of the information filtering system, is widely used in multiple industries. It excels at exploring and forecasting user preferences. In modern society, the rapid growth of E-Commerce business meaning that the recommender system will be an essential component for all online business platforms. For E-Commerce, it is a useful technique to help users make better decisions and enhance user loyalty. In the repository, we will proposed and implement a new model based on the exisiting autoencoder architecture. 

## Related Work 

Over the years, researchers have proposed different models and methods in tacklings recommender system, in collaborative filtering research area popular mthods includes item-based collaborative filterig and user-based collaborative filtering approach (Sarwar, B., Karypis, G., Konstan, J., & Riedl, J., 2001). 

In simple term, the main idea behind collaborative filtering is to generate predictions/ratings based on preferences of other similar user and items. In most cases, collaboaritve filtering has proven to make better recommendations compared to content based filtering. Moreover, the collaborative filerting method does not require the user to state their interest explicity (George, T., & Merugu, S. 2005). 

When talking about machine learning, KNN also known as Kth nearest neighbour will always be one of the first algorithm that you will learn, it is essentailly the "Hello World!" program when you start to learning programming. The same applies to recommender system, take user-based collaborative filtering for example, with the use of statisical techinique the model could find a set of users that shares similar interest. Though the algorithm is relatively easy to implement and able to ahieve good results, but it has a main disadvantage of curse of dimentionality which makes it hard to work with when number of the data is huge, more disucussion on this topic can be found here.(Sarwar, B., et al., 2001) (Su, X., & Khoshgoftaar, T., 2009) (Aggarwal, C., 2016)

In recent years, one of the algorithm that has received much traction is latent factor model, the model gained alot of popularity  after Netflix's pirze competition, the mentioned model includes SVD(Koren, Y., 2008), latent Dirichlet allocation (LDA) (Hofmann, T., 2004), pLSA (David, M.B., Andrew, N., & Michael, I.J., 2003), alternating least squares (ALS) (Jain, P., Netrapalli, P., & Sanghavi, S., 2013) are the matrix factorization method that has been proposed and explored in recommender system. SVD++ was later proposed and designed to estimate the implicit information, the result was that it surpasses both SVD and Asymmetric SVD (Koren, Y., 2008).

There is a lot of hype around deep learning lately, many different researchers have also came up with different deep learning models in exploring recommender system area. Neural Collaborative filtering (Xiangnan He, Lizi Liao, Hanwang Zhang et al, 2017) has proposed a feed-forward neural network that has since become a classical neural network method, other deep learning approach has also been
proposed such as restricted Boltzman machine (RBM) (Salakhutdinov, R., Mnih, A., & Hinton, G., 2007), deep autoencoder has also been explored in the past decade, because of how well autoencoder works in capturing important features. Different
variants of autoencoders has also been proposed, started of with AutoRec (Sedhain, S., Menon, A., Sanner, S., & Xie, L., 2015), stacked denoising auto encoder (Suzuki, Y., & Ozaki, T., 2017), Collaborative Variational autoencoder (Li, X., & She, J., 2017) where it seeks for probabilistic latent variable with the use of maximum a posteriori estimates(MAP).

## Methods

This time we will be looking at both NMF and autoencoders in collaborative filtering area, I am particularly interested in NMF after the implementation of [image reconstruction](https://github.com/MingSheng92/NMF), and I wanted to see if the same remains as effective when it is applied in a different challenge. For autoencoders, due to its similarity to NMF and being a part of deep learning so I would like to explore in this area. 


### Evaluation method 

To measure the performance of the algorithms, we decided to use the widely used evaluation metrics which are the Mean Absolute Error(MAE) and Root Mean Squared Error (RMSE). The defination of metrics is listed below:

 <a href="https://www.codecogs.com/eqnedit.php?latex=MAE&space;=&space;\frac{\sum_{(u,j)\in&space;E}&space;|e_{uj}|}{|E|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?MAE&space;=&space;\frac{\sum_{(u,j)\in&space;E}&space;|e_{uj}|}{|E|}" title="MAE = \frac{\sum_{(u,j)\in E} |e_{uj}|}{|E|}" /></a> , 
 <a href="https://www.codecogs.com/eqnedit.php?latex=RMSE&space;=&space;\sqrt{\frac{\sum_{(u,j)\in&space;E}&space;|e^2_{uj}|}{|E|}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?RMSE&space;=&space;\sqrt{\frac{\sum_{(u,j)\in&space;E}&space;|e^2_{uj}|}{|E|}}" title="RMSE = \sqrt{\frac{\sum_{(u,j)\in E} |e^2_{uj}|}{|E|}}" /></a>
 
Even though there is much more different evaluation methods, we will only use the above two for now.

### SVD

We have adopted the matrix factorization based algorithm (SVD), which is also equivalent to the probabilistic matrix factorization that has been proposed by(R.Salakhutdinov and A.Mnih.,2008). To predict the ratings of user to item i, we follow the equation that represents the relationship between average rating, user and item bias as well as the user and item interaction : 

<a href="https://www.codecogs.com/eqnedit.php?latex=R_{ui}&space;=&space;\mu&space;&plus;&space;b_u&space;&plus;&space;b_i&space;&plus;&space;q_i^TP_u" target="_blank"><img src="https://latex.codecogs.com/gif.latex?R_{ui}&space;=&space;\mu&space;&plus;&space;b_u&space;&plus;&space;b_i&space;&plus;&space;q_i^TP_u" title="R_{ui} = \mu + b_u + b_i + q_i^TP_u" /></a>

Where 𝝁 is the average rating, <a href="https://www.codecogs.com/eqnedit.php?latex=b_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b_i" title="b_i" /></a> & <a href="https://www.codecogs.com/eqnedit.php?latex=b_u" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b_u" title="b_u" /></a> is the user and item biased from global average, while is the vector that associated with each item i and each i user u associated with vector <a href="https://www.codecogs.com/eqnedit.php?latex=p_u" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_u" title="p_u" /></a> (Y. Koren, R. Bell, and C. Volinsky, 2009). To estimate the unknown ratings we also adopt the following regularized squared error: 

<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{r_{ui}&space;\in&space;R_{train}}&space;(R_{io}(True)&space;-&space;R_{ui})^2&space;&plus;&space;\lambda(b_i^2&space;&plus;&space;b_u^2&space;&plus;&space;\left&space;\|&space;q_i&space;\right&space;\|^2&space;&plus;&space;\left&space;\|&space;p_u&space;\right&space;\|^2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{r_{ui}&space;\in&space;R_{train}}&space;(R_{io}(True)&space;-&space;R_{ui})^2&space;&plus;&space;\lambda(b_i^2&space;&plus;&space;b_u^2&space;&plus;&space;\left&space;\|&space;q_i&space;\right&space;\|^2&space;&plus;&space;\left&space;\|&space;p_u&space;\right&space;\|^2)" title="\sum_{r_{ui} \in R_{train}} (R_{io}(True) - R_{ui})^2 + \lambda(b_i^2 + b_u^2 + \left \| q_i \right \|^2 + \left \| p_u \right \|^2)" /></a>

Then the minimization is performed with the following stochastic gradient descent: <br />
<a href="https://www.codecogs.com/eqnedit.php?latex=b_u&space;&\leftarrow&space;b_u&space;&&plus;&space;\gamma&space;(e_{ui}&space;-&space;\lambda&space;b_u)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b_u&space;&\leftarrow&space;b_u&space;&&plus;&space;\gamma&space;(e_{ui}&space;-&space;\lambda&space;b_u)" title="b_u &\leftarrow b_u &+ \gamma (e_{ui} - \lambda b_u)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=b_i&space;&\leftarrow&space;b_i&space;&&plus;&space;\gamma&space;(e_{ui}&space;-&space;\lambda&space;b_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?b_i&space;&\leftarrow&space;b_i&space;&&plus;&space;\gamma&space;(e_{ui}&space;-&space;\lambda&space;b_i)" title="b_i &\leftarrow b_i &+ \gamma (e_{ui} - \lambda b_i)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=p_u&space;&\leftarrow&space;p_u&space;&&plus;&space;\gamma&space;(e_{ui}&space;\cdot&space;q_i&space;-&space;\lambda&space;p_u)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p_u&space;&\leftarrow&space;p_u&space;&&plus;&space;\gamma&space;(e_{ui}&space;\cdot&space;q_i&space;-&space;\lambda&space;p_u)" title="p_u &\leftarrow p_u &+ \gamma (e_{ui} \cdot q_i - \lambda p_u)" /></a>

<a href="https://www.codecogs.com/eqnedit.php?latex=q_i&space;&\leftarrow&space;q_i&space;&&plus;&space;\gamma&space;(e_{ui}&space;\cdot&space;p_u&space;-&space;\lambda&space;q_i)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?q_i&space;&\leftarrow&space;q_i&space;&&plus;&space;\gamma&space;(e_{ui}&space;\cdot&space;p_u&space;-&space;\lambda&space;q_i)" title="q_i &\leftarrow q_i &+ \gamma (e_{ui} \cdot p_u - \lambda q_i)" /></a>

For more information on the algorithm, you can read the official documentation page of surprise in [SVD](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#)

### Autoencoder 

Autoencoder is widely used in denoising and decompression of image data, recently different variation of autoencoder has also been used as a generative model to generate sentences or even artwork (Jun-Yan Zhu, TaesungPark,Isola, P., & Efros, A. 2017).  
Similar to matrix factorization, autoencoder will try to learn the function : <br/>
<a href="https://www.codecogs.com/eqnedit.php?latex=h_{W,b}(x)\approx&space;x" target="_blank"><img src="https://latex.codecogs.com/gif.latex?h_{W,b}(x)\approx&space;x" title="h_{W,b}(x)\approx x" /></a>

In simple term, the model will try to learn an approximation and recontruct the input _x_, where it has two functions during training of the model, namely the encode function 
<a href="https://www.codecogs.com/eqnedit.php?latex=encode(x):&space;R^n&space;\rightarrow&space;R^d" target="_blank"><img src="https://latex.codecogs.com/gif.latex?encode(x):&space;R^n&space;\rightarrow&space;R^d" title="encode(x): R^n \rightarrow R^d" /></a> and the decode function <a href="https://www.codecogs.com/eqnedit.php?latex=decode(x):&space;R^d&space;\rightarrow&space;R^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?decode(x):&space;R^d&space;\rightarrow&space;R^n" title="decode(x): R^d \rightarrow R^n" /></a> to find the representation of reconstruction.

### Deep autoencoders with iterative refeeding feature 
In this repository, we are particularly interested in the autoencoders that is proposed by the cool people from NVIDIA where they have come up with an elegant deep learning autoencoder architechture with iterative refeeding (Kuchaiev, O., and Ginsburg, B., 2017).

Where their algorithm and refeeding steps can be defined as follows:

i. Given Sparse _x_, compute loss with Masked Mean Squared Error,where the x equation can be defined as: <br/>
<a href="https://www.codecogs.com/eqnedit.php?latex=MMSE&space;=&space;\frac{m_i&space;*&space;(r_i&space;-&space;y_i)}{\sum^{i=n}_{i=0}&space;m_i}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?MMSE&space;=&space;\frac{m_i&space;*&space;(r_i&space;-&space;y_i)}{\sum^{i=n}_{i=0}&space;m_i}" title="MMSE = \frac{m_i * (r_i - y_i)}{\sum^{i=n}_{i=0} m_i}" /></a>, <br />
where <a href="https://www.codecogs.com/eqnedit.php?latex=r_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r_i" title="r_i" /></a> is the actual rating and <a href="https://www.codecogs.com/eqnedit.php?latex=y_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y_i" title="y_i" /></a> is the predicted ratings, <a href="https://www.codecogs.com/eqnedit.php?latex=m_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m_i" title="m_i" /></a> is the masked ratings such that <a href="https://www.codecogs.com/eqnedit.php?latex=m_i" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m_i" title="m_i" /></a> = 1 if <a href="https://www.codecogs.com/eqnedit.php?latex=r_i&space;!=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r_i&space;!=&space;0" title="r_i != 0" /></a> else <a href="https://www.codecogs.com/eqnedit.php?latex=m_i&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m_i&space;=&space;0" title="m_i = 0" /></a>, and RMSE is just the square root of MMSE. 

ii. Update weights with back propagation.

iii. Reuse f(x) as new training example, and compute f(f(x)) with calculation of loss MMSE (refeeding feature). 

iv. update weights with back propagation.

Note that step 3 and step 4 can be performed multiple times during each iteration. 

<b>For more information on the paper and the mentioned model, you may refer to their respective [Github Repository](https://github.com/NVIDIA/DeepRecommender).</b>

### Deep Sparse autoencoders

Though that the model has produced decent result, I do believed that it was not well optimized, where it is expected to converge to expected error when large amount of data is used for training according to large number law. Secondly, the user item matrix generation will be an issue as the items increase which results in highly dependent on hardware resources to train the model. With the sparsity rate of the dataset that we are using, the model will not converge well even with a custom loss function. On the other hand, another paper that has proposed a similar model structure has come to conclusion that iterative refeeding with not bring any significant impact to the performance of the model as shown in FlexEncoderr (Tran, D.H., Hussain, Z., Zhang, W.E., Khoa, N. et al. 2019).

So insted of generating user and item matrix and feed into model to learn and optimize with custom loss function as mentioned earlier, we will use both user and item as embeddings then concatenate the two as new feature before feeding in back to the model. In order to further enforce the autoencoder to learn only the representation of the input data without redundancy in input, we will then apply regularization constraint to the autoencoder with L2 regularization, where the L2 term can be defined as below : 

<a href="https://www.codecogs.com/eqnedit.php?latex=L(x,&space;\hat&space;x)&space;&plus;&space;\lambda&space;\sum_i&space;a_i^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(x,&space;\hat&space;x)&space;&plus;&space;\lambda&space;\sum_i&space;a_i^2" title="L(x, \hat x) + \lambda \sum_i a_i^2" /></a>

The final proposed model can be seen in the model below: 

<img src="https://github.com/MingSheng92/RecommenderSystem/blob/master/img/test_keras_plot_model.png" data-canonical-src="https://github.com/MingSheng92/RecommenderSystem/blob/master/img/test_keras_plot_model.png" width="545" height="590" />

### Dataset

As the topic of interest is around online shopping, we wish to use a dataset that is closely related to the subject hence the final dataset chosen is none other than the Amazon dataset (McAuley, J., 2016). Dataset is selected based on the following criteria, dataset size has to be resaonably large, so that we can better stimulate a close to real-world e commerce environment for our experiment. Second, we will check if the database has a basic e-commerce platform’s attributes such as product data, user’s implicit and explicit ratings.

Other commonly used dataset for recommender system research includes, MovieLens (Grouplens, 2019), Brazillian E-Commerce Public Dataset by Olist(Kaggle, 2018), Netflix Prize dataset (Netflix, 2017), Retail rocket recommender system dataset (Retail rocket, 2017), Book-Crossing dataset (Cai-Nicolas, Z., 2004)

### Result 
<img src="https://github.com/MingSheng92/RecommenderSystem/blob/master/img/result.JPG" data-canonical-src="https://github.com/MingSheng92/RecommenderSystem/blob/master/img/result.JPG" width="424" height="107" />

### Reference
