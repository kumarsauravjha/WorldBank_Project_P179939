import folium
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm

# Sample data
edges = [
    ('San Francisco', 'Los Angeles', {'latlon': [(37.77, -122.41), (34.05, -118.24)], 'cost': 100}),
    ('San Francisco', 'Las Vegas', {'latlon': [(37.77, -122.41), (36.17, -115.14)], 'cost': 300}),
    ('Los Angeles', 'Las Vegas', {'latlon': [(34.05, -118.24), (36.17, -115.14)], 'cost': 150}),
]

# Create graph
G = nx.DiGraph()
for src, dest, attr in edges:
    G.add_edge(src, dest, **attr)

# Normalize costs for color map
costs = [attr['cost'] for _, _, attr in G.edges(data=True)]
norm = mcolors.Normalize(vmin=min(costs), vmax=max(costs))
cmap = cm.get_cmap('plasma')

# Create beautiful map
m = folium.Map(location=[36.5, -119.5], zoom_start=6, tiles="CartoDB positron")

# Add nodes as circle markers
locations = {
    'San Francisco': (37.77, -122.41),
    'Los Angeles': (34.05, -118.24),
    'Las Vegas': (36.17, -115.14),
}

for city, (lat, lon) in locations.items():
    folium.CircleMarker(
        location=(lat, lon),
        radius=7,
        color='black',
        fill=True,
        fill_color='blue',
        fill_opacity=0.7,
        popup=folium.Popup(f"<b>{city}</b>", max_width=200),
        tooltip=city
    ).add_to(m)

# Draw edges with color-graded cost
for src, dest, attr in G.edges(data=True):
    cost = attr['cost']
    color = mcolors.to_hex(cmap(norm(cost)))
    folium.PolyLine(
        locations=attr['latlon'],
        color=color,
        weight=4 + (cost / max(costs) * 4),  # dynamic width based on cost
        tooltip=f"{src} → {dest}<br>Cost: {cost}",
        popup=folium.Popup(f"<b>Route:</b> {src} → {dest}<br><b>Cost:</b> {cost}", max_width=250),
    ).add_to(m)

# Save map
m.save("graph_map.html")
