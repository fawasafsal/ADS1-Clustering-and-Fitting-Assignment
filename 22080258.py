import pandas as pd
import numpy as np
from collections import Counter
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Layer, InputSpec, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
import tensorflow as tf
from sklearn.manifold import TSNE

# Load the Netflix dataset
INPUT_DIR_NETFLIX = "netflix_titles.csv"
df_netflix_raw = pd.read_csv(INPUT_DIR_NETFLIX)

# Display some dataset properties
print("Some dataset properties:")
print(
    f"1.) The shape of the dataset is {df_netflix_raw.shape}, {df_netflix_raw.shape[0]} rows and {df_netflix_raw.shape[1]} columns")
print("-" * 80)
print("2.) The dataset columns contain the following datatypes:")
print(df_netflix_raw.dtypes)
print("-" * 80)
print("3.) NaN cells in the dataset:")
print(df_netflix_raw.isna().sum())
print("-" * 80)

# Check for duplicate titles and remove them
df_netflix = df_netflix_raw.drop_duplicates("title")
print(
    f"{df_netflix_raw.shape[0] - df_netflix.shape[0]} rows of duplicate titles have been removed")
print("-" * 80)

# Count number of unique genres
genres = pd.Series(", ".join(df_netflix.copy().fillna("")[
                   'listed_in']).split(", ")).unique().sum()
print(f"There are {len(genres)} unique categories/genres in this dataset")
print("-" * 80)

# Display information about movies and TV shows over the years
years = list(range(2008, 2020, 1))
movie_rows = df_netflix.loc[df_netflix["type"] == "Movie"]
tv_rows = df_netflix.loc[df_netflix["type"] == "TV Show"]
movies_counts = movie_rows.release_year.value_counts()
tv_counts = tv_rows.release_year.value_counts()
index_years_mov = movies_counts.index.isin(years)
index_years_tv = tv_counts.index.isin(years)
movies = movies_counts[index_years_mov]
tv_shows = tv_counts[index_years_tv]
movies_per = round(movie_rows.shape[0] / df_netflix["type"].shape[0] * 100, 2)
tvshows_per = round(tv_rows.shape[0] / df_netflix["type"].shape[0] * 100, 2)

# Top Movie and TV Show producer countries
top5_producer_countries = df_netflix.country.value_counts(
).sort_values(ascending=False).head(5)

# Top most common Actors and Directors (Movies and TV shows)
casts = ", ".join(df_netflix.copy().fillna("")['cast']).split(", ")
counter_list = Counter(casts).most_common(5)
most_common_actors = [i for i in counter_list if i[0] != ""]
labels_actors = [i[0] for i in most_common_actors][::-1]
values_actors = [i[1] for i in most_common_actors][::-1]

most_common_directors = df_netflix.director.value_counts().head(
    5).sort_values(ascending=True)

# Dashboard Creation of the dataset
width = 0.75
sns.set(style="whitegrid", palette="muted", color_codes=True)


def autolabel(rects, axes):
    for rect in rects:
        height = rect.get_height()
        axes.annotate('{}'.format(height),
                      xy=(rect.get_x() + rect.get_width() / 2, height),
                      xytext=(0, 3),  # 3 points vertical offset
                      textcoords="offset points",
                      ha='center', va='bottom')


# Set up the matplotlib figure
f, axes = plt.subplots(2, 2, figsize=(12, 12), sharex=False)

# Line plot of Movies and TV Shows released by Netflix per year
sns.lineplot(data=movies, color="b", ax=axes[0, 0], label="Movies / year")
sns.lineplot(data=tv_shows, color="c", ax=axes[0, 0], label="TV Shows / year")

# Pie chart of type percentages
axes[0, 1].pie([movies_per, tvshows_per], explode=(0, 0.1,), labels=["Movies", "TV Shows"], autopct='%1.1f%%',
               shadow=True, startangle=90)

# Bar chart of top 5 Movie / TV shows producer countries
rects1 = axes[1, 0].bar(top5_producer_countries.index,
                        top5_producer_countries.values)
autolabel(rects1, axes[1, 0])

# Bar chart of top 5 most common actors and directors
rects2 = axes[1, 1].bar(labels_actors, values_actors, width, label='Actors')
rects3 = axes[1, 1].bar(most_common_directors.index,
                        most_common_directors.values, width, label='Directors')
autolabel(rects2, axes[1, 1])
autolabel(rects3, axes[1, 1])

axes[0, 0].set_ylabel('Publications')
axes[0, 0].set_title('Movies / TV Shows released per year')

axes[0, 1].set_title('Percentage of Movies and TV Shows')

axes[1, 0].set_ylabel('Movies and TV Shows')
axes[1, 0].set_title('Top 5 producer countries')
axes[1, 0].legend()

axes[1, 1].set_ylabel('Number Occurring')
axes[1, 1].set_xticklabels(
    labels_actors + list(most_common_directors.index), rotation="vertical")
axes[1, 1].set_title('Top 5 most common actors and directors')

axes[0, 0].legend()
axes[0, 1].legend()

plt.tight_layout()
plt.savefig('output.png')
plt.show()

# Prepare data for clustering
df_token = df_netflix["description"]
maxlen = 1500
training_samples = 800
validation_samples = 450
max_words = 10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df_token)
sequences = tokenizer.texts_to_sequences(df_token)
word_index = tokenizer.word_index
print(f"{len(word_index)} unique tokens found")

data = pad_sequences(sequences, maxlen=maxlen)

scaler = MinMaxScaler()
x = scaler.fit_transform(data)

# Define autoencoder model


def autoencoder(dims, act='relu', init='glorot_uniform'):
    n_stacks = len(dims) - 1

    input_data = Input(shape=(dims[0],), name='input')
    x = input_data

    # internal layers of encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act,
                  kernel_initializer=init, name='encoder_%d' % i)(x)

    # latent hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init,
                    name='encoder_%d' % (n_stacks - 1))(x)

    x = encoded
    # internal layers of decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act,
                  kernel_initializer=init, name='decoder_%d' % i)(x)

    # decoder output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)

    decoded = x

    autoencoder_model = Model(
        inputs=input_data, outputs=decoded, name='autoencoder')
    encoder_model = Model(inputs=input_data, outputs=encoded, name='encoder')

    return autoencoder_model, encoder_model


# Call the autoencoder function to get the models
autoencoder, encoder = autoencoder([data.shape[-1], 500, 500, 1000, 10],
                                   init=VarianceScaling(scale=1. / 3., mode='fan_in', distribution='uniform'))

# Compile and fit the autoencoder model
autoencoder.compile(optimizer='rmsprop', loss='binary_crossentropy')
autoencoder.fit(x, x, batch_size=128, epochs=8)

# Define clustering layer


class ClusteringLayer(Layer):

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=tf.float32, shape=(None, input_dim))
        self.clusters = self.add_weight(name='clusters', shape=(
            self.n_clusters, input_dim), initializer='glorot_uniform')

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (tf.reduce_sum(tf.square(tf.expand_dims(inputs,
                   axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters


# Create clustering model
n_clusters = 20
clustering_layer = ClusteringLayer(
    n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)
model.compile(optimizer=SGD(0.01, 0.9), loss='kld')

# Fit KMeans and set initial weights of clustering layer
kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(x))
y_pred_last = np.copy(y_pred)
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# Compute target distribution


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


# Training loop
loss = 0
index = 0
maxiter = 1000
update_interval = 100
index_array = np.arange(x.shape[0])
tol = 0.001

# Add the missing batch_size definition
batch_size = 64

for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(x, verbose=0)
        # update the auxiliary target distribution p
        p = target_distribution(q)

    idx = index_array[index *
                      batch_size: min((index+1) * batch_size, x.shape[0])]
    loss = model.train_on_batch(x=x[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

# model.save_weights(save_dir + '/DEC_model_final.h5')

# Eval.
q = model.predict(x, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p

# evaluate the clustering performance
y_pred = q.argmax(1)

data_all = df_netflix.copy()
data_all['cluster'] = y_pred
data_all.head()
data_all['cluster'].value_counts()

x_embedded = TSNE(n_components=2).fit_transform(x)

# sns settings
sns.set(rc={'figure.figsize': (15, 15)})

# colors
palette = sns.color_palette("bright", len(set(y_pred)))

# plot
sns.scatterplot(x=x_embedded[:, 0], y=x_embedded[:, 1],
                hue=y_pred, legend='full', palette=palette)
plt.title("Netflix Movies and Tv Shows, Clustered (Autoencoder and Custom Keras Layer), Tf-idf with Plain Text")
plt.savefig('output2.png')
plt.show()

from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS
from bokeh.palettes import Category20
from bokeh.transform import linear_cmap
from bokeh.io import output_file, show
from bokeh.transform import transform
from bokeh.io import output_notebook
from bokeh.plotting import figure
from bokeh.layouts import column
from bokeh.models import RadioButtonGroup
from bokeh.models import TextInput
from bokeh.layouts import gridplot
from bokeh.models import Div
from bokeh.models import Paragraph

output_notebook()
y_labels = y_pred

# Data sources
source = ColumnDataSource(data=dict(
    x=x_embedded[:, 0],
    y=x_embedded[:, 1],
    x_backup=x_embedded[:, 0],
    y_backup=x_embedded[:, 1],
    desc=y_labels,
    titles=df_netflix['title'],
    directors=df_netflix['director'],
    cast=df_netflix['cast'],
    description=df_netflix['description'],
    labels=["C-" + str(x) for x in y_labels]
))

# Hover over information
hover = HoverTool(tooltips=[
    ("Title", "@titles"),
    ("Director(s)", "@directors"),
    ("Cast", "@cast"),
    ("Description", "@description"),
],
    point_policy="follow_mouse")

# Map colors
mapper = linear_cmap(field_name='desc',
                     palette=Category20[20],
                     low=min(y_labels), high=max(y_labels))

# Prepare the figure
p = figure(width=800, height=800,
           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'],
           title="Netflix Movies and Tv Shows, Clustered(Autoencoder and custom Keras Layer), Tf-idf with Plain Text",
           toolbar_location="right")

# Plot
p.scatter('x', 'y', size=5,
          source=source,
          fill_color=mapper,
          line_alpha=0.3,
          line_color="black")

# Option
option = RadioButtonGroup(labels=["C-0", "C-1", "C-2",
                                  "C-3", "C-4", "C-5",
                                  "C-6", "C-7", "C-8",
                                  "C-9", "C-10", "C-11",
                                  "C-12", "C-13", "C-14",
                                  "C-15", "C-16", "C-17",
                                  "C-18", "C-19", "All"],
                          active=20)

# Search box
# keyword = TextInput(title="Search:", callback=keyword_callback)
# Header
header = Div(text="""<h1>Similar movies / tv shown in corresponding to Cluster</h1>""")

# Show
show(column(header, p))