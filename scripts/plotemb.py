import matplotlib.pyplot as plt
import numpy as np

DPI = 100 # Dots per inch for plotting
outfile = 'cat_embeddings_heatmap.png'

def parse_line(line):
    filename, embedding_str = line.split('\t', 1)
    embedding = np.fromstring(embedding_str.strip()[1:-1], sep=',')
    return filename, embedding

def normalize_column(column): # Each column has its own min/max range for better visualization
    min_val = np.min(column)
    max_val = np.max(column)
    if max_val > min_val:     # Avoid division by zero
        return (column - min_val) / (max_val - min_val)
    else: 
        return np.zeros_like(column)

embeddings = []
with open('cat_embeddings_500.tsv', 'r') as f:
    for line in f:
        filename, embedding = parse_line(line)
        embeddings.append(embedding)

embeddings = np.array(embeddings)
norm_embeddings = np.apply_along_axis(normalize_column, 0, embeddings)
num_rows, num_columns = norm_embeddings.shape
print(f'Columns: {num_columns}, Rows: {num_rows}')

# Preserve 1:1 pixel mapping in output file + extra space for labels
fig_width_inch = num_columns / DPI
fig_height_inch = num_rows / DPI
fig, ax = plt.subplots(figsize=(fig_width_inch + 2, fig_height_inch + 2), dpi=DPI)
cax = ax.imshow(norm_embeddings, cmap='bwr', aspect='auto')

plt.colorbar(cax, label='Embedding Value')
ax.set_title('Embeddings Heatmap')
ax.set_xlabel('Embedding Index')
ax.set_ylabel('Cat Index')

plt.savefig(outfile, dpi=num_columns / fig_width_inch, bbox_inches='tight', pad_inches=0.5)
