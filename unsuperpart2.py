#Visualizing the PCA transformation

#instructions
'''Import:
matplotlib.pyplot as plt.
pearsonr from scipy.stats.
Assign column 0 of grains to width and column 1 of grains to length.
Make a scatter plot with width on the x-axis and length on the y-axis.
Use the pearsonr() function to calculate the Pearson correlation of width and length.'''


# Perform the necessary imports
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Assign the 0th column of grains: width
width = grains[:,0]

# Assign the 1st column of grains: length
length = grains[:,1]

# Scatter plot width vs length
plt.scatter(width, length)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation
correlation, pvalue = pearsonr(width,length)

# Display the correlation
print(correlation)


#instructions
'''Import PCA from sklearn.decomposition.
Create an instance of PCA called model.
Use the .fit_transform() method of model to apply the PCA transformation to grains. Assign the result to pca_features.
The subsequent code to extract, plot, and compute the Pearson correlation of the first two columns pca_features has been written for you, so hit submit to see the result!'''

# Import PCA
from sklearn.decomposition import PCA

# Create PCA instance: model
model = PCA()

# Apply the fit_transform method of model to grains: pca_features
pca_features = model.fit_transform(grains)

# Assign 0th column of pca_features: xs
xs = pca_features[:,0]

# Assign 1st column of pca_features: ys
ys = pca_features[:,1]

# Scatter plot xs vs ys
plt.scatter(xs, ys)
plt.axis('equal')
plt.show()

# Calculate the Pearson correlation of xs and ys
correlation, pvalue = pearsonr(xs, ys)

# Display the correlation
print(correlation)


#Intrinsic Dimension
#instructions
'''Make a scatter plot of the grain measurements. This has been done for you.
Create a PCA instance called model.
Fit the model to the grains data.
Extract the coordinates of the mean of the data using the .mean_ attribute of model.
Get the first principal component of model using the .components_[0,:] attribute.
Plot the first principal component as an arrow on the scatter plot, using the plt.arrow() function. You have to specify the first two arguments - mean[0] and mean[1].'''

# Make a scatter plot of the untransformed points
plt.scatter(grains[:,0], grains[:,1])

# Create a PCA instance: model
model = PCA()

# Fit model to points
model.fit(grains)

# Get the mean of the grain samples: mean
mean = model.mean_

# Get the first principal component: first_pc
first_pc = model.components_[0,:]

# Plot first_pc as an arrow, starting at mean
plt.arrow(mean[0], mean[1], first_pc[0], first_pc[1], color='red', width=0.01)

# Keep axes on same scale
plt.axis('equal')
plt.show()


#instructions
'''Create an instance of StandardScaler called scaler.
Create a PCA instance called pca.
Use the make_pipeline() function to create a pipeline chaining scaler and pca.
Use the .fit() method of pipeline to fit it to the fish samples samples.
Extract the number of components used using the .n_components_ attribute of pca. Place this inside a range() function and store the result as features.
Use the plt.bar() function to plot the explained variances, with features on the x-axis and pca.explained_variance_ on the y-axis.'''

# Perform the necessary imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

# Create scaler: scaler
scaler = StandardScaler()

# Create a PCA instance: pca
pca = PCA()

# Create pipeline: pipeline
pipeline = make_pipeline(scaler, pca)

# Fit the pipeline to 'samples'
pipeline.fit(samples)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.xticks(features)
plt.show()


#Dimension reduction with PCA
#instructions
'''mport PCA from sklearn.decomposition.
Create a PCA instance called pca with n_components=2.
Use the .fit() method of pca to fit it to the scaled fish measurements scaled_samples.
Use the .transform() method of pca to transform the scaled_samples. Assign the result to pca_features.'''

# Import PCA
from sklearn.decomposition import PCA

# Create a PCA model with 2 components: pca
pca = PCA(n_components = 2)

# Fit the PCA instance to the scaled samples
pca.fit(scaled_samples)

# Transform the scaled samples: pca_features
pca_features =pca.transform(scaled_samples)

# Print the shape of pca_features
print(pca_features.shape)



#instructions
'''Import TfidfVectorizer from sklearn.feature_extraction.text.
Create a TfidfVectorizer instance called tfidf.
Apply .fit_transform() method of tfidf to documents and assign the result to csr_mat. This is a word-frequency array in csr_matrix format.
Inspect csr_mat by calling its .toarray() method and printing the result. This has been done for you.
The columns of the array correspond to words. Get the list of words by calling the .get_feature_names() method of tfidf, and assign the result to words.'''

# Import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer: tfidf
tfidf = TfidfVectorizer() 

# Apply fit_transform to document: csr_mat
csr_mat = tfidf.fit_transform(documents)

# Print result of toarray() method
print(csr_mat.toarray())

# Get the words: words
words = tfidf.get_feature_names()

# Print words
print(words)



#instructions
'''Import:
TruncatedSVD from sklearn.decomposition.
KMeans from sklearn.cluster.
make_pipeline from sklearn.pipeline.
Create a TruncatedSVD instance called svd with n_components=50.
Create a KMeans instance called kmeans with n_clusters=6.
Create a pipeline called pipeline consisting of svd and kmeans.'''

# Perform the necessary imports
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Create a TruncatedSVD instance: svd
svd = TruncatedSVD(n_components= 50)

# Create a KMeans instance: kmeans
kmeans = KMeans(n_clusters = 6)

# Create a pipeline: pipeline
pipeline = make_pipeline(svd,kmeans)



#instructions
'''Import pandas as pd.
Fit the pipeline to the word-frequency array articles.
Predict the cluster labels.
Align the cluster labels with the list titles of article titles by creating a DataFrame df with labels and titles as columns. This has been done for you.
Use the .sort_values() method of df to sort the DataFrame by the 'label' column, and print the result.
Hit submit and take a moment to investigate your amazing clustering of Wikipedia pages!'''

# Import pandas
import pandas as pd

# Fit the pipeline to articles
pipeline.fit(articles)

# Calculate the cluster labels: labels
labels = pipeline.predict(articles)

# Create a DataFrame aligning labels and titles: df
df = pd.DataFrame({'label': labels, 'article': titles})

# Display df sorted by cluster label
print(df.sort_values('label'))



#Non-negative matrix factorization 
#instructions
'''Import NMF from sklearn.decomposition.
Create an NMF instance called model with 6 components.
Fit the model to the word count data articles.
Use the .transform() method of model to transform articles, and assign the result to nmf_features.
Print nmf_features to get a first idea what it looks like (.round(2) rounds the entries to 2 decimal places.)'''

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF instance: model
model = NMF(n_components = 6)

# Fit the model to articles
model.fit(articles)

# Transform the articles: nmf_features
nmf_features = model.transform(articles)

# Print the NMF features
print(nmf_features.round(2))



#instructions
'''Import pandas as pd.
Create a DataFrame df from nmf_features using pd.DataFrame(). Set the index to titles using index=titles.
Use the .loc[] accessor of df to select the row with title 'Anne Hathaway', and print the result. These are the NMF features for the article about the actress Anne Hathaway.
Repeat the last step for 'Denzel Washington' (another actor).'''

# Import pandas
import pandas as pd

# Create a pandas DataFrame: df
df = pd.DataFrame(nmf_features, index = titles)

# Print the row for 'Anne Hathaway'
print(df.loc["Anne Hathaway"])

# Print the row for 'Denzel Washington'
print(df.loc['Denzel Washington'])


#instructions
'''Import pandas as pd.
Create a DataFrame components_df from model.components_, setting columns=words so that columns are labeled by the words.
Print components_df.shape to check the dimensions of the DataFrame.
Use the .iloc[] accessor on the DataFrame components_df to select row 3. Assign the result to component.
Call the .nlargest() method of component, and print the result. This gives the five words with the highest values for that component.'''

# Import pandas
import pandas as pd

# Create a DataFrame: components_df
components_df = pd.DataFrame(model.components_, columns = words)

# Print the shape of the DataFrame
print(components_df.shape)

# Select row 3: component
component = components_df.iloc[3]

# Print result of nlargest
print(component.nlargest())


#instructions
'''Import matplotlib.pyplot as plt.
Select row 0 of samples and assign the result to digit. For example, to select column 2 of an array a, you could use a[:,2]. Remember that since samples is a NumPy array, you can't use the .loc[] or iloc[] accessors to select specific rows or columns.
Print digit. This has been done for you. Notice that it is a 1D array of 0s and 1s.
Use the .reshape() method of digit to get a 2D array with shape (13, 8). Assign the result to bitmap.
Print bitmap, and notice that the 1s show the digit 7!
Use the plt.imshow() function to display bitmap as an image.'''

# Import pyplot
from matplotlib import pyplot as plt

# Select the 0th row: digit
digit = samples[0,:]

# Print digit
print(digit)

# Reshape digit to a 13x8 array: bitmap
bitmap = digit.reshape(13,8)

# Print bitmap
print(bitmap)

# Use plt.imshow to display bitmap
plt.imshow(bitmap, cmap='gray', interpolation='nearest')
plt.colorbar()
plt.show()


#instructions
'''Import NMF from sklearn.decomposition.
Create an NMF instance called model with 7 components. (7 is the number of cells in an LED display).
Apply the .fit_transform() method of model to samples. Assign the result to features.
To each component of the model (accessed via model.components_), apply the show_as_image() function to that component inside the loop.
Assign the row 0 of features to digit_features.
Print digit_features.'''

# Import NMF
from sklearn.decomposition import NMF

# Create an NMF model: model
model = NMF(n_components = 7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)

# Select the 0th row of features: digit_features
digit_features = features[0,:]

# Print digit_features
print(digit_features)


#instructions
'''Import PCA from sklearn.decomposition.
Create a PCA instance called model with 7 components.
Apply the .fit_transform() method of model to samples. Assign the result to features.
To each component of the model (accessed via model.components_), apply the show_as_image() function to that component inside the loop.'''

# Import PCA
from sklearn.decomposition import PCA

# Create a PCA instance: model
model = PCA(n_components = 7)

# Apply fit_transform to samples: features
features = model.fit_transform(samples)

# Call show_as_image on each component
for component in model.components_:
    show_as_image(component)
    
    
# Building recommender system using NMF  
#instructions
'''Import normalize from sklearn.preprocessing.
Apply the normalize() function to nmf_features. Store the result as norm_features.
Create a DataFrame df from norm_features, using titles as an index.
Use the .loc[] accessor of df to select the row of 'Cristiano Ronaldo'. Assign the result to article.
Apply the .dot() method of df to article to calculate the cosine similarity of every row with article.
Print the result of the .nlargest() method of similarities to display the most similar articles. This has been done for you, so hit 'Submit Answer' to see the result!'''

# Perform the necessary imports
import pandas as pd
from sklearn.preprocessing import normalize

# Normalize the NMF features: norm_features
norm_features = normalize(nmf_features)

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index = titles)

# Select the row corresponding to 'Cristiano Ronaldo': article
article = df.loc['Cristiano Ronaldo']

# Compute the dot products: similarities
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())


#instructions
'''Import:
NMF from sklearn.decomposition.
Normalizer and MaxAbsScaler from sklearn.preprocessing.
make_pipeline from sklearn.pipeline.
Create an instance of MaxAbsScaler called scaler.
Create an NMF instance with 20 components called nmf.
Create an instance of Normalizer called normalizer.
Create a pipeline called pipeline that chains together scaler, nmf, and normalizer.
Apply the .fit_transform() method of pipeline to artists. Assign the result to norm_features.'''

# Perform the necessary imports
from sklearn.decomposition import NMF
from sklearn.preprocessing import Normalizer, MaxAbsScaler 
from sklearn.pipeline import make_pipeline

# Create a MaxAbsScaler: scaler
scaler = MaxAbsScaler()

# Create an NMF model: nmf
nmf = NMF(n_components = 20)

# Create a Normalizer: normalizer
normalizer = Normalizer()

# Create a pipeline: pipeline
pipeline = make_pipeline(scaler, nmf, normalizer)

# Apply fit_transform to artists: norm_features
norm_features = pipeline.fit_transform(artists)



#instructions
'''mport pandas as pd.
Create a DataFrame df from norm_features, using artist_names as an index.
Use the .loc[] accessor of df to select the row of 'Bruce Springsteen'. Assign the result to artist.
Apply the .dot() method of df to artist to calculate the dot product of every row with artist. Save the result as similarities.
Print the result of the .nlargest() method of similarities to display the artists most similar to 'Bruce Springsteen'.'''

# Import pandas
import pandas as pd

# Create a DataFrame: df
df = pd.DataFrame(norm_features, index = artist_names)

# Select row of 'Bruce Springsteen': artist
artist = df.loc['Bruce Springsteen']

# Compute cosine similarities: similarities
similarities = df.dot(artist)

# Display those with highest cosine similarity
print(similarities.nlargest())
