import os
os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import pandas as pd
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem, Circle
from bokeh.palettes import Category20
from sklearn.cluster import KMeans
import json
from pathlib import Path

pn.extension()

class UMAPVisualizer:
    def __init__(self, umap_file_path="umap_embeddings.npy"):
        self.data = None
        self.umap_embedding = None
        self.source = None
        self.plot = None
        self.n_clusters = 10
        self.umap_file_path = umap_file_path
        self.classifications_file = None
        self.cluster_colors = None
        self.sample_size = 50000
        self.use_sampling = True
        self.full_data = None  # Full dataset
        
        # Create widgets
        self.cluster_slider = pn.widgets.IntSlider(
            name='Number of Clusters', 
            start=5, end=50, value=10, 
            width=250
        )
        
        # Add sampling controls
        self.sample_toggle = pn.widgets.Toggle(
            name='Sample Datapoints for Display', 
            value=True,
            width=200
        )
        
        self.sample_slider = pn.widgets.IntSlider(
            name='Sample Size', 
            start=10000, end=100000, value=50000, step=10000,
            width=250
        )
        
        self.cluster_button = pn.widgets.Button(
            name='Update Clustering',
            button_type='default',
            width=200
        )
        
        # Classification buttons
        self.classify_cell_button = pn.widgets.Button(
            name='Mark Selected as CELL', 
            button_type='primary',
            width=200
        )
        
        self.classify_not_cell_button = pn.widgets.Button(
            name='Mark Selected as NOT CELL', 
            button_type='primary',
            width=200
        )
        
        self.reset_button = pn.widgets.Button(
            name='Reset Classifications', 
            button_type='default',
            width=200
        )
        
        self.save_button = pn.widgets.Button(
            name='Save Classifications',
            button_type='success', 
            width=200
        )
        
        # Status text
        self.status_text = pn.pane.Markdown(f"**Status:** Loading {umap_file_path}...", width=400)
        
        # Set up callbacks
        self.cluster_button.on_click(self.update_clusters)
        self.classify_cell_button.on_click(self.classify_as_cell)
        self.classify_not_cell_button.on_click(self.classify_as_not_cell)
        self.reset_button.on_click(self.reset_classifications)
        self.save_button.on_click(self.save_classifications)
        self.sample_toggle.param.watch(self.on_sample_toggle, 'value')
        self.sample_slider.param.watch(self.on_sample_size_change, 'value')
        self.current_legend = None  # Add this line
        
        # Initialize empty plot
        self.create_empty_plot()
        
        # Automatically load data at startup
        self.load_umap_data()
    
    def on_sample_toggle(self, event):
        """Handle sample toggle changes"""
        self.use_sampling = event.new
        if self.full_data is not None:
            self.prepare_display_data()
            self.update_plot()
    
    def on_sample_size_change(self, event):
        """Handle sample size changes"""
        self.sample_size = event.new
        if self.use_sampling and self.full_data is not None:
            self.prepare_display_data()
            self.update_plot()
    
    def load_umap_data(self):
        """Load UMAP embeddings from .npy file"""
        try:
            if not os.path.exists(self.umap_file_path):
                self.status_text.object = f"**Error:** File not found: {self.umap_file_path}"
                return
            
            # Load the numpy array
            self.umap_embedding = np.load(self.umap_file_path)
            
            if self.umap_embedding.ndim != 2 or self.umap_embedding.shape[1] != 2:
                self.status_text.object = "**Error:** UMAP file must be 2D with shape (n_points, 2)"
                return
                
            n_points = self.umap_embedding.shape[0]
            
            # Set up classifications file path
            file_stem = Path(self.umap_file_path).stem
            self.classifications_file = os.path.join(os.getcwd(), "curation", f"{file_stem}_classifications.json")

            # Initialize or load existing classifications
            self.load_existing_classifications(n_points)
            
            # Generate cluster colors
            self.cluster_colors = Category20[20]
            
            # Perform initial clustering
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
            initial_clusters = kmeans.fit_predict(self.umap_embedding)
            
            # Store full dataset
            self.full_data = pd.DataFrame({
                'umap_x': self.umap_embedding[:, 0],
                'umap_y': self.umap_embedding[:, 1],
                'cluster': initial_clusters,
                'classification': self.classifications,
                'original_index': np.arange(n_points)  # Track original indices
            })
            
            # Determine if sampling is needed
            if n_points > 100000:
                self.use_sampling = True
                self.sample_toggle.value = True
                self.status_text.object = f"**Status:** Large dataset ({n_points:,} points) - sampling enabled for performance"
            else:
                self.use_sampling = False
                self.sample_toggle.value = False
            
            # Prepare display data
            self.prepare_display_data()
            
            # Update plot WITH legend (initial load)
            self.update_plot(update_legend=True)
            
            display_points = len(self.data)
            self.status_text.object = f"**Status:** Loaded {n_points:,} points, displaying {display_points:,} with {self.n_clusters} clusters"
                
        except Exception as e:
            self.status_text.object = f"**Error:** Could not load UMAP file: {str(e)}"
    
    def prepare_display_data(self):
        """Prepare data for display (with optional sampling)"""
        if self.full_data is None:
            return
            
        if self.use_sampling and len(self.full_data) > self.sample_size:
            # Sample data for display
            sample_indices = np.random.choice(len(self.full_data), self.sample_size, replace=False)
            self.data = self.full_data.iloc[sample_indices].copy().reset_index(drop=True)
        else:
            self.data = self.full_data.copy()
    
    def load_existing_classifications(self, n_points):
        """Load existing classifications if available"""
        self.classifications = ['unclassified'] * n_points
        
        if os.path.exists(self.classifications_file):
            try:
                with open(self.classifications_file, 'r') as f:
                    saved_data = json.load(f)
                    if len(saved_data['classifications']) == n_points:
                        self.classifications = saved_data['classifications']
                        self.status_text.object = f"**Status:** Loaded existing classifications"
                    else:
                        self.status_text.object = f"**Warning:** Classification file size mismatch, starting fresh"
            except Exception as e:
                self.status_text.object = f"**Warning:** Could not load classifications: {str(e)}"
    
    def save_classifications(self, event):
        """Save current classifications to file"""
        if self.full_data is None or self.classifications_file is None:
            return
            
        try:
            # Use full dataset for saving, not just the displayed sample
            full_classifications = self.full_data['classification'].tolist()
            
            cell_count = sum(1 for c in full_classifications if c == 'cell')
            not_cell_count = sum(1 for c in full_classifications if c == 'not_cell')
            unclassified_count = sum(1 for c in full_classifications if c == 'unclassified')
            
            classifications_data = {
                'filename': self.umap_file_path,
                'n_points': len(self.full_data),
                'classifications': full_classifications,
                'counts': {
                    'cell': cell_count,
                    'not_cell': not_cell_count, 
                    'unclassified': unclassified_count
                }
            }
            
            with open(self.classifications_file, 'w') as f:
                json.dump(classifications_data, f, indent=2)
            
            self.status_text.object = f"**Status:** Saved - Cells: {cell_count:,}, Not cells: {not_cell_count:,}, Unclassified: {unclassified_count:,}"
            
        except Exception as e:
            self.status_text.object = f"**Error:** Could not save classifications: {str(e)}"
    
    def create_empty_plot(self):
        """Create empty plot optimized for large datasets"""
        self.plot = figure(
            width=700, height=500,
            title="",
            tools="pan,wheel_zoom,box_zoom,box_select,lasso_select,reset,save",
            toolbar_location="right",
            output_backend="webgl"  # Use WebGL for much better performance
        )
        
        # Style the plot
        self.plot.xaxis.axis_label = "UMAP Dimension 1"
        self.plot.yaxis.axis_label = "UMAP Dimension 2"
        
        # Optimize plot settings for performance
        self.plot.toolbar.autohide = True
    
    def get_point_colors_and_properties_fast(self):
        """Optimized color assignment using vectorized operations"""
        n_points = len(self.data)
        
        # Pre-allocate arrays
        colors = np.empty(n_points, dtype=object)
        alphas = np.full(n_points, 0.4)
        
        # Get classification and cluster arrays
        classifications = self.data['classification'].values
        clusters = self.data['cluster'].values
        
        # Vectorized color assignment
        cell_mask = classifications == 'cell'
        not_cell_mask = classifications == 'not_cell'
        unclassified_mask = ~(cell_mask | not_cell_mask)
        
        # Set colors
        colors[cell_mask] = 'black'
        colors[not_cell_mask] = 'gray'
        
        # For unclassified, use cluster colors
        for i in np.where(unclassified_mask)[0]:
            colors[i] = self.cluster_colors[clusters[i] % len(self.cluster_colors)]
        
        # Set alphas
        alphas[cell_mask] = 0.8
        alphas[not_cell_mask] = 0.05
        
        return colors.tolist(), alphas.tolist()
    
    def update_plot(self, update_legend=False):
        """Update the plot with optimized rendering"""
        if self.data is None:
            return
        
        # Get colors efficiently
        colors, alphas = self.get_point_colors_and_properties_fast()
        
        # Prepare data source with minimal data
        self.source = ColumnDataSource(data=dict(
            x=self.data['umap_x'].values,
            y=self.data['umap_y'].values,
            cluster=self.data['cluster'].values,
            classification=self.data['classification'].values,
            colors=colors,
            alphas=alphas
        ))
        
        # Clear existing renderers
        self.plot.renderers = []
        
        # Simplified hover tool (only when needed)
        if len(self.data) < 50000:  # Only add hover for smaller datasets
            hover = HoverTool(tooltips=[
                ("Cluster", "@cluster"),
                ("Classification", "@classification")
            ])
            self.plot.toolbar.tools = [tool for tool in self.plot.toolbar.tools if not isinstance(tool, HoverTool)]
            self.plot.add_tools(hover)
        
        # Create optimized scatter plot
        self.scatter = self.plot.scatter(
            'x', 'y', 
            source=self.source,
            size=3,  # Smaller points for performance
            color='colors',
            alpha='alphas',
            line_color=None,  # Remove line borders for performance
            selection_color="red",
            nonselection_alpha=0.1
        )
        
        if update_legend==True:
            # currently no legend but leaving this arg in for future use
            pass

    
    def update_clusters(self, event=None):
        """Update clustering with performance optimizations"""
        if self.full_data is None or self.umap_embedding is None:
            return
            
        self.n_clusters = self.cluster_slider.value
        self.status_text.object = f"**Status:** Computing {self.n_clusters} clusters, please wait..."
        
        try:
            # Cluster on full dataset
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
            new_clusters = kmeans.fit_predict(self.umap_embedding)
            
            # Update full dataset
            self.full_data['cluster'] = new_clusters
            
            # Update display data
            self.prepare_display_data()
            
            # Update plot
            self.update_plot(update_legend=True)
            
            self.status_text.object = f"**Status:** Updated to {self.n_clusters} clusters"
            
        except Exception as e:
            self.status_text.object = f"**Error:** Clustering failed: {str(e)}"
    
    def classify_as_cell(self, event):
        self._classify_selected('cell')
    
    def classify_as_not_cell(self, event):
        self._classify_selected('not_cell')
    
    def _classify_selected(self, classification_type):
        """Classify selected points with proper index mapping"""
        if self.source is None:
            return
            
        selected_indices = self.source.selected.indices
        
        if not selected_indices:
            self.status_text.object = "**Status:** No points selected. Use lasso or box select first."
            return
        
        # Map display indices back to original indices if using sampling
        if self.use_sampling and 'original_index' in self.data.columns:
            original_indices = self.data.iloc[selected_indices]['original_index'].values
            self.full_data.loc[original_indices, 'classification'] = classification_type
        else:
            self.full_data.loc[selected_indices, 'classification'] = classification_type
        
        # Update display data
        for idx in selected_indices:
            self.data.loc[idx, 'classification'] = classification_type
        
        # Update plot
        self.update_plot(update_legend=False)

        # Clear selection
        self.source.selected.indices = []
        
        # Count from full dataset
        cell_count = sum(1 for c in self.full_data['classification'] if c == 'cell')
        not_cell_count = sum(1 for c in self.full_data['classification'] if c == 'not_cell')
        
        self.status_text.object = f"**Status:** Classified {len(selected_indices):,} points as '{classification_type}' | Total - Cells: {cell_count:,}, Not cells: {not_cell_count:,}"
    
    def reset_classifications(self, event):
        """Reset all classifications"""
        if self.full_data is None:
            return
            
        self.full_data['classification'] = 'unclassified'
        self.prepare_display_data()
        self.update_plot(update_legend=False)

        self.status_text.object = "**Status:** Reset all classifications"
    
    def get_layout(self):
        """Return the Panel layout"""
        plot_pane = pn.pane.Bokeh(self.plot, sizing_mode='stretch_width')
        
        plot_title = pn.pane.Markdown("### Data Curation UMAP", width=700, margin=(0, 0, 10, 0))
        
        stats_display = pn.pane.Markdown("", width=700, margin=(0, 0, 10, 0))
        
        if self.full_data is not None:
            total_points = len(self.full_data)
            display_points = len(self.data) if self.data is not None else 0
            cell_count = sum(1 for c in self.full_data['classification'] if c == 'cell')
            not_cell_count = sum(1 for c in self.full_data['classification'] if c == 'not_cell')
            
            if self.use_sampling:
                stats_display.object = f"**Total:** {total_points:,} points | **Displaying:** {display_points:,} (sampled) | **Cells:** {cell_count:,} | **Not cells:** {not_cell_count:,}"
            else:
                stats_display.object = f"**{total_points:,} points** | **Cells:** {cell_count:,} | **Not cells:** {not_cell_count:,}"
        
        plot_column = pn.Column(
            plot_title,
            stats_display,
            plot_pane,
            width=720
        )
        
        controls = pn.Column(
            "## Performance Controls",
            self.sample_toggle,
            self.sample_slider,
            pn.Spacer(height=20),
            "## Clustering Controls",
            self.cluster_slider,
            pn.Spacer(height=10),
            self.cluster_button,
            pn.pane.Markdown("*Adjust slider then click 'Update Clustering'*", width=300),
            pn.Spacer(height=20),
            "## Classification Controls", 
            self.classify_cell_button,
            pn.Spacer(height=10),
            self.classify_not_cell_button,
            pn.Spacer(height=10),
            self.reset_button,
            pn.Spacer(height=10),
            self.save_button,
            pn.Spacer(height=20),
            self.status_text,
            width=420,
            margin=(10, 10)
        )
        
        return pn.Row(
            plot_column,
            pn.Spacer(width=20),
            controls,
            sizing_mode='stretch_width'
        )

# Create the application
def create_app(umap_file="umap_embeddings.npy"):
    """Create the UMAP visualization app optimized for large datasets"""
    visualizer = UMAPVisualizer(os.path.join(os.getcwd(), "curation", umap_file))
    return visualizer.get_layout()

# Create and serve the app
app = create_app("umap_2d.npy")
app.servable()

if __name__ == "__main__":
    app.show(port=5007)