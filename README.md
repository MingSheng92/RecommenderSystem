# Work in progress


## RecommenderSystem

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

 <a href="https://www.codecogs.com/eqnedit.php?latex=MAE&space;=&space;\frac{\sum_{(u,j)\in&space;E}&space;|e_{uj}|}{|E|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?MAE&space;=&space;\frac{\sum_{(u,j)\in&space;E}&space;|e_{uj}|}{|E|}" title="MAE = \frac{\sum_{(u,j)\in E} |e_{uj}|}{|E|}" /></a>
 
 <a href="https://www.codecogs.com/eqnedit.php?latex=RMSE&space;=&space;\sqrt{\frac{\sum_{(u,j)\in&space;E}&space;|e^2_{uj}|}{|E|}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?RMSE&space;=&space;\sqrt{\frac{\sum_{(u,j)\in&space;E}&space;|e^2_{uj}|}{|E|}}" title="RMSE = \sqrt{\frac{\sum_{(u,j)\in E} |e^2_{uj}|}{|E|}}" /></a>
 
 Even though there is much more different evaluation methods, we will only use the above two for now.
