#!/usr/bin/env python
# coding: utf-8

# In[43]:


import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.datasets import make_regression, make_classification, load_diabetes, load_breast_cancer, load_wine, load_digits 
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import bias_variance_decomp
from sklearn.ensemble import BaggingRegressor, BaggingClassifier, RandomForestRegressor 
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree 
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1


# In[ ]:


n_sample = 1000
sample = np.random.normal(0,1,n_sample) 
bootstrap = 1000
moyenne = []
def test(bootstrap):
    for i in range(1,bootstrap+1):
        sample_b = np.random.choice(sample,size=n_sample,replace = True)
        moyenne.append(np.mean(sample_b))

    return sample_b


# In[ ]:


b_500 = test(500)
b_250 = test(250)
b_1000 = test(1000)
normal = sample


# In[ ]:


sns.set_style("whitegrid")
fig, axs = plt.subplots(2, 2, figsize=(10, 6))
sns.distplot(normal, ax=axs[0, 0])
sns.distplot(b_250, ax=axs[0, 1])
sns.distplot(b_500, ax=axs[1, 0])
sns.distplot(b_1000, ax=axs[1, 1])


axs[0, 0].set_title("Echantillon de base N(0,1)")
axs[0, 1].set_title("Echantillon bootstrap avec 250 éléments")
axs[1, 0].set_title("Echantillon bootstrap avec 500 éléments")
axs[1, 1].set_title("Echantillon bootstrap avec 1000 éléments")



plt.savefig('bootstrap_memoire_1.png',format ='png')
plt.show()


# In[ ]:


X_diab, y_diab = load_diabetes(return_X_y=True)
X_diab_train, X_diab_test, y_diab_train, y_diab_test = train_test_split(X_diab,y_diab,test_size = 0.2,random_state=42)


# In[ ]:


X_cancer, y_cancer = load_breast_cancer(return_X_y=True)
X_cancer_train, X_cancer_test, y_cancer_train, y_cancer_test = train_test_split(X_cancer,y_cancer,test_size = 0.2,random_state=42)


# In[ ]:


boot = [10*i for i in range(1,11)]
linear = GaussianNB()
#avg_loss_c, avg_bias_c, avg_var_c = bias_variance_decomp(linear,X_diab_train,y_cancer_train,X_cancer_test,y_cancer_test,loss='0-1_loss',
                                                   #random_seed = 123)


# In[ ]:



#bagg = BaggingClassifier(base_estimator=linear,n_estimators=100)
#avg_lossc1, avg_biasc1, avg_varc1 = bias_variance_decomp(bagg,X_cancer_train,y_cancer_train,X_cancer_test,y_cancer_test,loss='0-1_loss',
                                                        #random_seed = 123)


# In[ ]:


classf = LogisticRegression()
classf.fit(X_cancer_train,y_cancer_train)
y_cancer_pred = classf.predict(X_cancer_test)
acc = accuracy_score(y_cancer_test,y_cancer_pred)
print((1-acc) * 100)


# In[ ]:


classf1 = BaggingClassifier(base_estimator = classf, n_estimators = 100)
classf1.fit(X_cancer_train,y_cancer_train)
y_cancer_pred1 = classf1.predict(X_cancer_test)
acc1 = accuracy_score(y_cancer_test,y_cancer_pred1)
print(acc1)


# In[ ]:


######print("Average Loss:", avg_loss_c)
#####print("Average Bias:", avg_bias_c)
####print("Average Variance:", avg_var_c)
###print("Average Loss:", avg_lossc1)
##print("Average Bias:", avg_biasc1)
#print("Average Variance:", avg_varc1)


# In[ ]:


#y_predict = bagg.predict(X_diab_test)
#print(y_predict)


# In[ ]:


#plt.plot(X_diab_test,y_predict)
#plt.show()


# In[ ]:


print((1-acc1) * 100)


# In[ ]:


#données vin 
X_wine, y_wine = load_wine(return_X_y=True)
X_wine_train,X_wine_test, y_wine_train, y_wine_test = train_test_split(X_wine,y_wine,test_size=0.2, random_state=42)


# In[ ]:


#données digits
X_digits, y_digits = load_digits(return_X_y=True)
X_digits_train,X_digits_test, y_digits_train, y_digits_test = train_test_split(X_digits,y_digits,test_size=0.2, random_state=42)


# In[ ]:


classf2 = LogisticRegression()
classf2.fit(X_wine_train,y_wine_train)
y_wine_pred2 = classf2.predict(X_wine_test)
acc2 = accuracy_score(y_wine_test,y_wine_pred2)
print((1-acc2) * 100)

classf3 = BaggingClassifier(base_estimator = LogisticRegression(),n_estimators = 100)
classf3.fit(X_wine_train,y_wine_train)
y_wine_pred3 = classf3.predict(X_wine_test)
acc3 = accuracy_score(y_wine_test,y_wine_pred3)
print((1-acc3) * 100)


# In[ ]:


print((1-acc2) * 100)
print((1-acc3) * 100)


# In[ ]:


classf3 = LogisticRegression()
classf3.fit(X_digits_train,y_digits_train)
y_digits_pred3 = classf3.predict(X_digits_test)
acc3 = accuracy_score(y_digits_test,y_digits_pred3)
print((1-acc3) * 100)

classf4 = BaggingClassifier(base_estimator = LogisticRegression(),n_estimators = 100)
classf4.fit(X_digits_train,y_digits_train)
y_digits_pred4 = classf4.predict(X_digits_test)
acc4 = accuracy_score(y_digits_test,y_digits_pred4)
print((1-acc4) * 100)


# In[ ]:


print((1-acc3) * 100)
print((1-acc4) * 100)


# In[ ]:



def rule_function(x):
    qualitative_rule = x[0] == 1
    
    quantitative_rules = [
        x[1] > 0.5,
        x[2] < 0.2
    ]
    
    if qualitative_rule and all(quantitative_rules):
        return 1
    else:
        return 0

#nombre de données
n_observations = 1000

# Generer les variables qualitatives
qualitative_feature = np.random.randint(2, size=n_observations)

# Generer les variables quantitatives
quantitative_features = np.random.rand(n_observations, 2)

features = np.concatenate((qualitative_feature.reshape(-1, 1), quantitative_features), axis=1)
labels = np.array([rule_function(x) for x in features])

# ajouter du bruit aux variables quantitatives
quantitative_noise_scale = 0.1
quantitative_features += np.random.normal(loc=0, scale=quantitative_noise_scale, size=quantitative_features.shape)

# séparer le train set et le test set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

#  Entrainer l'arbre
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)

# regarder les résultats
y_pred = tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# plot l'arbre de décision 
plt.figure(figsize=(10, 6))
plot_tree(tree,feature_names=["Qualitative", "Quantitative 1", "Quantitative 2"], class_names=["0", "1"], filled=True)

plt.show()


# In[ ]:


avg_loss, avg_bias, avg_var = bias_variance_decomp(tree,X_train,y_train,X_test,y_test,loss='0-1_loss',
                                                    random_seed = 123)


# In[ ]:


print(avg_loss)
print(avg_var)
print(avg_bias)


# In[ ]:



np.random.seed(42)
x = np.linspace(-10, 10, 100)
noise = np.random.normal(0, 15, size=100)
y = 3*x**2 + 2*x + 5 + noise


x = x.reshape(-1, 1)


reg_tree = DecisionTreeRegressor()
reg_tree.fit(x, y)
reg_forest = RandomForestRegressor(n_estimators=1000)
reg_forest.fit(x,y)


y_pred = reg_tree.predict(x)
y_pred_forest = reg_forest.predict(x)
 
sorted_indices = np.argsort(x.flatten())
x_sorted = x[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]
y_pred_forest_sorted = y_pred_forest[sorted_indices]



fig, (ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
ax1.scatter(x, y, label='Points')
ax1.plot(x_sorted, y_pred_sorted, color='red', label='Arbre de régression')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Arbre de régression')
ax1.legend()

ax2.scatter(x, y, label='Points')
ax2.plot(x_sorted, y_pred_forest_sorted, color='red', label='Fôret aléatoire')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Forêt aléatoire')
ax2.legend()
plt.savefig('comparaison_arbre_foret.png',format='png')
plt.show()


# In[ ]:



np.random.seed(42)
x = np.linspace(-10, 10, 100)
noise = np.random.normal(0, 15, size=100)
y = 3*x**2 + 2*x + 5 + noise


x = x.reshape(-1, 1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



reg_tree = DecisionTreeRegressor()
reg_tree.fit(x_train, y_train)


y_pred = reg_tree.predict(x_test)
loss, bias, var = bias_variance_decomp(reg_tree,x_train,y_train, x_test,  y_test,loss='mse')

path = reg_tree.cost_complexity_pruning_path(x_train,y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
mse = []
for i in ccp_alphas: 
    arbre = DecisionTreeRegressor(ccp_alpha=i)
    arbre.fit(x_train,y_train)
    _,_,vari = bias_variance_decomp(arbre,x_train,y_train,x_test,y_test,loss='mse')
    mse.append(vari)
depth = [1,2,3,4,5,6,7,8,9,10]
vct = []
b = []
e = []
for dep in depth:
    arbre = DecisionTreeRegressor(max_depth=dep)
    arbre.fit(x_train,y_train)
    err,bias,vari = bias_variance_decomp(arbre,x_train,y_train,x_test,y_test,loss='mse')
    vct.append(vari)
    b.append(bias)
    e.append(err)



sorted_indices = np.argsort(x_test.flatten())
x_test_sorted = x_test[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]


plt.scatter(x_train, y_train, label='Training Data')
plt.scatter(x_test, y_test, color='orange', label='Test Data')
plt.plot(x_test_sorted, y_pred_sorted, color='red', label='Regression Tree')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

print("loss of the regression tree on the test set:", loss)
print("bias of the regression tree on the test set:",bias)
print("variance of the regression tree on the test set:", var)


# In[ ]:


plt.plot(depth,e)
plt.plot(depth,vct)
plt.plot(depth,b)
plt.show()


# In[ ]:


plt.plot(ccp_alphas,mse)
plt.xlabel('alpha')
plt.ylabel('variance')
plt.title('Variance en fonction de alpha')
plt.savefig('variance_alpha.png',format='png')
plt.show()


# In[39]:



np.random.seed(42)
x = np.linspace(-10, 10, 100)
noise = np.random.normal(0, 15, size=100)
y = 3*x**2 + 2*x + 5 + noise


x = x.reshape(-1, 1)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


rf_regressor = RandomForestRegressor(n_estimators = 1000)
rf_regressor.fit(x_train, y_train)


y_pred = rf_regressor.predict(x_test)
loss, bias, var = bias_variance_decomp(rf_regressor,x_train,y_train, x_test,  y_test,loss='mse')
bt = [100,200,500,1000]
err =[]
biais = []
vari = []
for i in bt: 
    forest = RandomForestRegressor(n_estimators=i)
    loss,bias,var = bias_variance_decomp(forest,x_train,y_train,x_test,y_test,loss='mse')
    err.append(loss)
    biais.append(bias)
    vari.append(var)




sorted_indices = np.argsort(x_test.flatten())
x_test_sorted = x_test[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]


plt.scatter(x_train, y_train, label='Training Data')
plt.scatter(x_test, y_test, color='orange', label='Test Data')
plt.plot(x_test_sorted, y_pred_sorted, color='red', label='Random Forest Regression')
plt.xlabel('x')
plt.ylabel('y')
#plt.title('Random Forest Regression (Variance: {:.2f})'.format(variance))
plt.legend()
plt.show()

print("loss of the regression tree on the test set:", loss)
print("bias of the regression tree on the test set:",bias)
print("variance of the regression tree on the test set:", var)


# In[42]:


plt.plot(bt,err,label='Erreur quadratique')
plt.plot(bt,biais,label='Biais^2')
plt.plot(bt,vari,label='variance')
plt.xlabel('Nombre de bootstrap')
plt.ylabel('Erreur décomposée')
plt.legend()
plt.savefig('erreur_decomp.png',format='png')
plt.show()


# In[ ]:




