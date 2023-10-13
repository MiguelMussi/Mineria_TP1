''' TP MINERÍA DE DATOS'''

# Enlace al notebook con las ejecuciones y visualizaciones
# https://colab.research.google.com/drive/1pxpMOUctDN0go9OrFGLCxU4NOK6RToSt

# Nota: Instalación previa de recursos necesarios

# Importacion de librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn import decomposition
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
import umap
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
from hdbscan import HDBSCAN



''' DATAFRAME '''
df = pd.read_csv('Crop_recommendation.csv', sep=',', engine='python')
print(df.head())


''' ANÁLISIS EXPLORATORIO '''
print(df.info())

# --- Recategorización de datos 
# Se convierte el tipo de "label" a categoría.
df['label'] = df['label'].astype('category')

# --- Análisis de datos nulos y faltantes
print(df.isna().sum())
# No existen datos nulos ni faltantes


# --- Balance de datos
# Se analiza la distribución equitativa de las clases "label"
# Agrupar por etiqueta y contar la frecuencia
lbl_frec = df['label'].value_counts()

# Crear el gráfico de barras
plt.figure(figsize=(6, 4))
lbl_frec.plot(kind='bar')
plt.title('Frecuencia por label')
plt.xlabel('Label')
plt.ylabel('Frecuencia')
plt.xticks(rotation=90, ha='center')
plt.tight_layout()
plt.show()

# Comprobación de cantidad de registros por label
print(lbl_frec)


# --- Análisis estadístico del DF
print(df.describe())


# --- Histogramas
df.hist(figsize=(20,10))
plt.show()


# --- Boxplot por 'labels' variables

# Variable a analizar
target = 'temperature'
# Configuración de estilo y tamaño del gráfico
plt.figure(figsize=(10, 6))
# Crear el gráfico de boxplot con barrios en el eje vertical y precios en el eje horizontal
boxplot = plt.boxplot([df[df['label'] == label][target] for label in df['label'].unique()],
                      vert=False,  # Boxplots horizontales
                      patch_artist=True)  # Para personalizar colores de los cuadros
# Personalizar el color de los cuadros (box) y los bigotes (whisker)
for box in boxplot['boxes']:
    box.set(facecolor='lightblue')
for whisker in boxplot['whiskers']:
    whisker.set(color='gray', linewidth=1.2, linestyle='--')
# Etiquetas y título del gráfico
plt.yticks(range(1, len(df['label'].unique()) + 1), df['label'].unique())  # Etiquetas en el eje y
plt.xlabel(target)
plt.ylabel('Label')
plt.title('Boxplot de categorías por Label')
# Desactivar notación científica en el eje horizontal (precios)
plt.ticklabel_format(axis='x', style='plain')
# Mostrar el gráfico
plt.tight_layout()
plt.show()



''' DATOS DE INTERÉS '''
# Cultivos con condiciones extremas
print("Algunas observaciones interesantes")
print("---------------------------------")
print("Cultivos que requieren una proporción muy alta de Nitrógeno:", df[df['N'] > 120]['label'].unique())
print("Cultivos que requieren una proporción muy alta de Fósforo:", df[df['P'] > 100]['label'].unique())
print("Cultivos que requieren una proporción muy alta de Potasio:", df[df['K'] > 200]['label'].unique())
print("Cultivos que requieren mucha lluvia:", df[df['rainfall'] > 200]['label'].unique())
print("Cultivos que requieren muy bajas temperaturas:", df[df['temperature'] < 10]['label'].unique())
print("Cultivos que requieren muy altas temperaturas :", df[df['temperature'] > 40]['label'].unique())
print("Cultivos que requieren muy baja humedad:", df[df['humidity'] < 20]['label'].unique())
print("Cultivos que requieren un ph muy bajo:", df[df['ph'] < 4]['label'].unique())
print("Cultivos que requieren un ph muy alto:", df[df['ph'] > 9]['label'].unique())

# Cultivos estacionales
print("Cultivos de Verano")
print(df[(df['temperature'] > 30) & (df['humidity'] > 50)]['label'].unique())
print("-----------------------------------")
print("Cultivos de Invierno")
print(df[(df['temperature'] < 20) & (df['humidity'] > 30)]['label'].unique())
print("-----------------------------------")
print("Cultivos de lluvia")
print(df[(df['rainfall'] > 200) & (df['humidity'] > 30)]['label'].unique())



''' CORRELACIONES ENTRE ATRIBUTOS '''
# --- Matriz de corelaciones
corr = df.drop(['label'], axis=1).corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot = True,
    annot_kws = {'size': 6}
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=0,
    horizontalalignment='center'
)
plt.show()



''' ESTANDARIZACIÓN '''

# --- Estandarización Z-score por cálculo
df_sub1 = df.drop(['label'], axis=1)
df_std1 = (df_sub1-df_sub1.mean())/df_sub.std()

# --- Estandarización Z-score por librería
df_sub = df.drop(['label'], axis=1) # Eliminación de variables no numéricas
scaler = StandardScaler() # Creación del objeto scaler
X_scaled = scaler.fit_transform(df_sub) # Cálculo de la media y la desviación estándar y aplicación de la transformación de estandarización.
df_std = pd.DataFrame(X_scaled, columns=df_sub.columns)



''' PCA '''

# --- Aplicación de la técnica
# Obtener todas las componentes principales
pca = PCA(n_components=df_sub.shape[1])
pca_features = pca.fit_transform(df_std)

# PC dataframe
pca_df = pd.DataFrame(
    data=pca_features,
    columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'])
pca_df['label'] = df['label']

# Eigenvectors
pd.DataFrame(pca.components_, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7'], index=['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7'])

# --- Selección del número de componentes principales
# Función para acumular la varianza
def var_acum(numbers):
     sum = 0
     var_c = []
     for num in numbers:
        sum += num
        var_c.append(sum)
     return var_c

var_c = var_acum(pca.explained_variance_ratio_)
pca_rtd = pd.DataFrame({'Eigenvalues':pca.explained_variance_, 'Proporción de variancia explicada':pca.explained_variance_ratio_, 'Proporción acumulada de variancia explicada': var_c})

# Vemos gráficamente la varianza acumulada:
wish = 0.75
plt.bar(range(1, 8), pca.explained_variance_ratio_,
        alpha=0.5,
        align='center')
# modificado para reemplazar el escalonado por la curva
plt.plot(range(1, 8), np.cumsum(pca.explained_variance_ratio_),
         color='red')
# agregada línea horizontal para un valor determinado
plt.axhline(y=wish, color='gray', linestyle='--')
plt.ylabel('Proporción de varianza explicada')
plt.xlabel('Componentes principales')
plt.show()

# --- Criterios de selección
# * Proporción de variancia acumulada (~75% -80%)​
# * Criterio de Kaiser (eigenvalues > 1)​
# * Gráfico del codo (Scree)

''' Las tres primeras componentes acumulan el 65% ~ 70% de la variabilidad total,
 es decir, están cercanas a cumplir con el primer criterio (>~75%). 
 Si se consideraran las componentes cuyos eigenvalues son superiores a 1 
 (Criterio de Kaiser) se debería optar por extraer cuatro. 
 Por conveniencia y practicidad para graficar las distribuciones 
 de las componentes se decide seleccionar tres de ellas (primer citerio)'''

# --- Gráfico del codo
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='steelblue')
plt.title('Scree Plot')
plt.xlabel('Componentes principales')
plt.ylabel('Proporción de variancia explicada')
plt.show()

'''Al observar el gráfico del codo, vemos que el quiebre parece producirse 
entre la segunda y tercera componente. 
Considerando la primera y la segunda componentes llegaríamos a un ~60% 
de la variabilidad total, por lo que se considera óptimo tomar hasta 
la tercera componente.'''


# --- Matriz de correlación de PC seleccionados
corr = pca_df[['PC1', 'PC2', 'PC3']].corr()
ax = sns.heatmap(
    corr,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True,
    annot = True,
    annot_kws = {'size': 6}
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=0,
    horizontalalignment='center'
)
plt.show()


# --- Scatter Plots
df.drop(columns=['label']).columns.to_list()
features = df.drop(columns=['label']).columns.to_list()

loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
fig = px.scatter(pca_features, x=0, y=1, color = pca_df["label"],  labels={'color': 'label'} )
fig.update_layout(title = "Biplot",width = 1200,height = 600)
fig.show()
fig = px.scatter_3d(pca_features, x=0, y=1, z=2,
              color=pca_df["label"],  labels={'color': 'label'})
fig.show()



''' ISOMAP '''

# --- Aplicación de la técnica
# Gráfico
isomap_df = Isomap(n_neighbors=2, n_components=3)
isomap_df.fit(df_std)
projections_isomap = isomap_df.transform(df_std)

fig = px.scatter_3d(
    projections_isomap, x=0, y=1, z=2,
    color=df['label'], labels={'color': 'label'}
)
fig.update_traces(marker_size=8)
fig.show()



''' T-SNE '''

# --- Aplicación de la técnica
# Gráfico
tsne = TSNE(n_components=3, random_state=0, perplexity=5)
projections_tsne = tsne.fit_transform(df_std, )

fig = px.scatter_3d(
    projections_tsne, x=0, y=1, z=2,
    color=df['label'], labels={'color': 'label'}
)
fig.update_traces(marker_size=8)
fig.show()



''' UMAP'''

# --- Aplicación de la técnica
# Gráfico
umap_2d = umap.UMAP(n_components=2, init='random', random_state=0)
umap_3d = umap.UMAP(n_components=3, init='random', random_state=0)

proj_2d = umap_2d.fit_transform(df_std)
proj_3d = umap_3d.fit_transform(df_std)

fig_2d = px.scatter(
    proj_2d, x=0, y=1,
    color=df['label'], labels={'color': 'label'}
)
fig_3d = px.scatter_3d(
    proj_3d, x=0, y=1, z=2,
    color=df['label'], labels={'color': 'label'}
)
fig_3d.update_traces(marker_size=8)

fig_2d.show()
fig_3d.show()



''' K-MEANS '''

# --- Método del codo
# Cálculo de la inercia
inercia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_scaled)
    inercia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inercia, marker='o')
plt.title('Método del Codo')
plt.xlabel('Número de Clústeres')
plt.ylabel('Inercia')
plt.xticks(np.arange(1, 11))
plt.grid(True)
plt.show()


# --- GAP Statistics
def calculate_intra_cluster_dispersion(X_scaled, k, linkage='ward'):
    clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = clustering.fit_predict(X_scaled)
    centroids = np.array([np.mean(X_scaled[labels == i], axis=0) for i in range(k)])
    intra_cluster_dispersion = np.sum(np.linalg.norm(X_scaled[labels] - centroids[labels], axis=1)**2)
    return intra_cluster_dispersion

gaps = []
max_k = 15
for k in range(1, max_k + 1):
    real_inertia = calculate_intra_cluster_dispersion(X_scaled, k, linkage='ward')

    inertia_list = []
    for _ in range(10):
      random_data = np.random.rand(*X_scaled.shape)
      intra_cluster_dispersion = calculate_intra_cluster_dispersion(random_data, k)
      inertia_list.append(intra_cluster_dispersion)

    reference_inertia = np.mean(inertia_list)

    gap = np.log(reference_inertia) - np.log(real_inertia)
    gaps.append(gap)

optimal_k = np.argmax(gaps) + 1

print("Número óptimo de clusters según el Gap Statistic:", optimal_k)

plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k + 1), gaps, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Gap Statistic')
plt.title('Gap Statistic para determinar el número óptimo de clusters (Clustering Jerárquico)')
plt.show()


# --- Aplicación de la técnica
# Creación del modelo
kmeans = KMeans(n_clusters=4)
kmeans.fit(X_scaled) #Entrenamos el modelo
# El metodo labels_ nos da a que cluster corresponde cada observacion
df['Cluster KMeans'] = kmeans.labels_


# Otra forma de comprobar cultivos por clusters
x = df.loc[:, ['N','P','K','temperature','ph','humidity','rainfall']].values
# Implementación de K-Means
km = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_means = km.fit_predict(x)
# Resultados
a = df['label']
y_means = pd.DataFrame(y_means)
z = pd.concat([y_means, a], axis = 1)
z = z.rename(columns = {0: 'cluster'})

# Cultivos por cluster
print("Análisis de cultivos por cluster \n")
print("Crops in First Cluster:", z[z['cluster'] == 0]['label'].unique())
print("---------------------------------------------------------------")
print("Crops in Second Cluster:", z[z['cluster'] == 1]['label'].unique())
print("---------------------------------------------------------------")
print("Crops in Third Cluster:", z[z['cluster'] == 2]['label'].unique())
print("---------------------------------------------------------------")
print("Crops in Fourth Cluster:", z[z['cluster'] == 3]['label'].unique())


# --- Centroides
# caracteristicas normalizadas que tendria el centroide de ese cluster.
print(kmeans.cluster_centers_)


# --- Cantidad de observaciones por cluster
observaciones_por_cluster = df['Cluster KMeans'].value_counts().sort_index()
print(observaciones_por_cluster)


# Gráfico de observaciones por cluster
df.groupby('Cluster KMeans').sum().plot(kind='bar', figsize=(10, 6))
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


# Relaciones entre clusters
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

plt.sca(ax[0])
for cluster_label in df['Cluster KMeans'].unique():
    cluster_data = df[df['Cluster KMeans'] == cluster_label]
    plt.scatter(cluster_data['P'], cluster_data['K'], label=f'Cluster {cluster_label}', alpha=0.5)

plt.xlabel('P - Fósforo')
plt.ylabel('K - Potasio')
plt.title('Relación entre Fósforo y Potasio por Clúster')
plt.legend()

plt.sca(ax[1])
for cluster_label in df['Cluster KMeans'].unique():
    cluster_data = df[df['Cluster KMeans'] == cluster_label]
    plt.scatter(cluster_data['temperature'], cluster_data['humidity'], label=f'Cluster {cluster_label}', alpha=0.5)

plt.xlabel('Temperatura')
plt.ylabel('Humedad')
plt.title('Relación entre Temperatura y Humedad por Clúster')
plt.legend()
plt.tight_layout()
plt.show()


# Comprobación de la distribución de los clusters mediante PCA
pca = PCA(n_components=4)
componentes_principales = pca.fit_transform(X_scaled)

# Gráfico 2D de las PC 
plt.figure(figsize=(10, 6))
plt.scatter(componentes_principales[:, 0], componentes_principales[:, 1], c=df['Cluster KMeans'] ,cmap='rainbow', alpha=0.5)
plt.xlabel('Primera Componente Principal')
plt.ylabel('Segunda Componente Principal')
plt.title('Visualización de Clústeres utilizando PCA con 2 componentes principales')
plt.legend()
plt.show()


# Gráfico 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['P'], df['K'], df['temperature'], c=df['Cluster KMeans'])
# Etiqueta los ejes
ax.set_xlabel('P - Fósforo')
ax.set_ylabel('k - Potasio')
ax.set_zlabel('Temperatura')
plt.show()



''' CLUSTERING JERÁRQUICO '''

# --- Coeficiente de Silhuette
def calculate_silhouette(X_scaled, k, linkage='ward'):
    clustering = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = clustering.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, labels)
    sample_silhouette_values = silhouette_samples(X_scaled, labels)
    return silhouette_avg, sample_silhouette_values

max_k = 15

silhouette_scores = []
for k in range(2, max_k + 1):
    silhouette_avg, _ = calculate_silhouette(X_scaled, k)
    silhouette_scores.append(silhouette_avg)


# Gráfico de coeficiente de Silhuette
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_k + 1), silhouette_scores, marker='o')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Coeficiente de Silhouette')
plt.title('Coeficiente de Silhouette para determinar el número óptimo de clusters (Clustering Jerárquico)')
plt.show()


# --- Aplicación de la técnica
# Dendograma completo
Z = linkage(X_scaled, "ward")
dendrogram(Z)
plt.show()

# Dendograma truncado
dendrogram(Z,  truncate_mode = 'lastp', p = 20, show_leaf_counts = False, show_contracted = True)
plt.axhline(y=30, c='k', linestyle='dashed')
plt.show()


# --- Método del codo para clustering jerárquico
# Distancias
distancias=[]
for i in range(1, 30):
    clustering = AgglomerativeClustering(n_clusters=i)
    clustering.fit(X_scaled)

    # Calculo la matriz de distancias entre puntos
    pairwise_distances = cdist(X_scaled, X_scaled, 'euclidean')

    # Calculo la distancia total entre los clusters
    distancia_total = 0
    for j in range(i):
        cluster_indices = np.where(clustering.labels_ == j)
        distancia_total += pairwise_distances[cluster_indices][:, cluster_indices].sum()

    distancias.append(distancia_total)

# Gráfico
plt.plot(range(1, 30), distancias, marker='o')
plt.xlabel('Número de Clusters')
plt.ylabel('Distancia Total')
plt.title('Método del Codo para Clustering Jerárquico')
plt.show()


# Asignación de clusters
n_clusters = 4
clustering = AgglomerativeClustering(n_clusters=n_clusters)
cluster_assignments = clustering.fit_predict(X_scaled)
df['Cluster'] = cluster_assignments
print(df['Cluster'].value_counts())


# Silhuette score
from sklearn.metrics import silhouette_score,silhouette_samples
silhouette_avg = silhouette_score(X_scaled, cluster_assignments)
print(silhouette_avg)


# --- DBSCAN y HDBSCAN
# DBSCAN
dbscan = DBSCAN(eps=1, min_samples=100)
dbscan_labels = dbscan.fit_predict(X_scaled)
#Elimino duplicados para ver cuantos grupos tengo y veo la cantidad que tengo
groups = len(set(dbscan_labels))
#Evaluo cual de ellos es ruido (si es -1)
count_noise = (1 if -1 in dbscan_labels else 0)
print("Número de clústeres identificados por DBSCAN:", groups - count_noise)
df['Cluster DBSCAN'] = dbscan_labels


# HDBSCAN
hdbscan = HDBSCAN(min_cluster_size=100)
hdbscan_labels = hdbscan.fit_predict(X_scaled)
print("Número de clústeres identificados por HDBSCAN:", len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0))
df['Cluster HDBSCAN'] = hdbscan_labels



