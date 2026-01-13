import numpy as np
import panel as pn
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, TapTool, ColorBar, LinearColorMapper
from bokeh.palettes import Category20, Reds256, Inferno256, Viridis256
from bokeh.colors import RGB

class UMAPVisualiser:
    """Pure UMAP visualisation component - handles only plot rendering and interactions"""
    
    def __init__(self, data=None, shot_noise=None, footprint_size=None, edge_cells=None, session_id=None):
        """
        Initialize with data DataFrame containing: umap_x, umap_y, cluster, classification
        """
        self.data = data
        self.shot_noise = shot_noise
        self.footprint_size = footprint_size
        self.edge_cells = edge_cells
        self.session_id = session_id
        self.probs = None
        self.source = None
        self.plot = None
        self.scatter = None
        self.cluster_colors = Category20[20]
        self.view_mode = 'clus'
        self.last_clicked_cluster = None
        self.prob_mapper = LinearColorMapper(palette=Reds256[::-1], low=0, high=1)
        self.color_bar = None
        
        # Callback for cluster selection - set by orchestrator
        self.on_cluster_selected = None
        
        # Create plot
        self.create_empty_plot()
        
        # Initial render if data provided
        if self.data is not None:
            self.update_plot()

    def set_probs(self, probs):
        """Set probabilities for coloring points"""
        if self.data is not None and probs.shape[0] != self.data.shape[0]:
            print(self.data.shape, probs.shape)
            raise ValueError("Length of probabilities must match number of data points.")
        self.probs = probs

        if self.data is not None:
            self.update_plot()
    
    def set_view_mode(self, event):
        """Set view mode: 'clus', 'prob', 'snr', or 'size'"""
        mode = event.new.lower()
        if mode not in ['clus', 'prob', 'snr', 'size', 'session', 'edge']:
            raise ValueError("View mode must be 'clus', 'prob', 'snr', 'session', 'edge', or 'size'.")
        
        self.view_mode = mode
        self.update_plot()

    def update_data(self, new_data):
        """Update with new data and re-render"""
        self.data = new_data
        self.view_mode = 'clus'  # Reset to cluster mode on data update
        self.update_plot()
    
    def get_plot_pane(self):
        """Return Panel pane containing the plot"""
        return pn.pane.Bokeh(self.plot, sizing_mode='stretch_width')
    
    def create_empty_plot(self):
        """Create empty plot optimized for large datasets"""
        base_tools = "pan,wheel_zoom,box_zoom,box_select,lasso_select,reset,save"
        
        self.plot = figure(
            width=700, height=500,
            title="",
            tools=base_tools,
            toolbar_location="right",
            output_backend="webgl"  # Use WebGL for better performance
        )
        
        # Style the plot
        self.plot.xaxis.axis_label = "UMAP Dimension 1"
        self.plot.yaxis.axis_label = "UMAP Dimension 2"
        
        # Optimize plot settings for performance
        self.plot.toolbar.autohide = True
        
        # Set up initial tools
        self.update_plot_tools()
    
    def update_plot_tools(self):
        """Update plot tools based on current mode"""
        if self.plot is None:
            return
        
        # Clear existing tools and rebuild
        self.plot.toolbar.tools = []
        
        # Add basic navigation tools
        from bokeh.models import PanTool, WheelZoomTool, BoxZoomTool, ResetTool, SaveTool
        self.plot.add_tools(PanTool(), WheelZoomTool(), BoxZoomTool(), ResetTool(), SaveTool())
        
        # Add tap tool for cluster selection
        tap_tool = TapTool()
        self.plot.add_tools(tap_tool)
    
    def on_point_tap(self, attr, old, new):
        """Handle point tap in cluster mode"""
        if not new or self.data is None:
            return
        
        # Get the cluster of the tapped point
        tapped_idx = new[0]
        if tapped_idx < len(self.data):
            self.last_clicked_cluster = self.data.iloc[tapped_idx]['cluster']
            
            # Notify orchestrator if callback is set
            if self.on_cluster_selected:
                self.on_cluster_selected(self.last_clicked_cluster, tapped_idx)
    
    def get_point_colors_and_properties(self):
        """Color assignment using vectorized operations"""
        if self.data is None or len(self.data) == 0:
            return [], []
        
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
        
        if self.view_mode == "prob" and self.probs is not None:
            # For all ROIs, use probabilities to set colors
            for i in range(n_points):
                prob = self.probs[i]
                colors[i] = RGB(r=255, g=int(255 * (1 - prob)), b=int(255 * (1 - prob)))
        elif self.view_mode == "snr":
            # For all ROIs, use shot noise to set colors
            min_noise = np.min(self.shot_noise)
            max_noise = np.max(self.shot_noise)
            norm_noise = (self.shot_noise - min_noise) / (max_noise - min_noise + 1e-8)
            for i in range(n_points):
                noise_val = norm_noise[i]
                colors[i] = Inferno256[int(noise_val * 255)]
        elif self.view_mode == "size":
            # For all ROIs, use footprint size to set colors
            min_size = np.min(self.footprint_size)
            max_size = np.max(self.footprint_size)
            norm_size = (self.footprint_size - min_size) / (max_size - min_size + 1e-8)
            for i in range(n_points):
                size_val = norm_size[i]
                colors[i] = Viridis256[int(size_val * 255)]
        elif self.view_mode == "session":
            # For all ROIs, use session ID to set colors
            unique_sessions = np.unique(self.session_id)
            session_color_map = {sess: Category20[20][i % 20] for i, sess in enumerate(unique_sessions)}
            for i in range(n_points):
                sess_id = self.session_id[i]
                colors[i] = session_color_map[sess_id]
        elif self.view_mode == "edge":
            # For all ROIs, use edge cell status to set colors
            for i in range(n_points):
                if self.edge_cells[i]:
                    colors[i] = 'red'
                else:
                    colors[i] = 'blue'
        else:
            # Cluster mode
            colors[cell_mask] = 'black'
            colors[not_cell_mask] = 'gray'
            # For unclassified, use cluster colors
            for i in np.where(unclassified_mask)[0]:
                colors[i] = self.cluster_colors[clusters[i] % len(self.cluster_colors)]
        
        # Set alphas
        alphas[cell_mask] = 0.8
        alphas[not_cell_mask] = 0.05
        
        return colors.tolist(), alphas.tolist()
    
    def update_plot(self):
        """Update the plot with current data"""
        if self.data is None or len(self.data) == 0:
            return
        
        # Get colors efficiently
        colors, alphas = self.get_point_colors_and_properties()
        
        # Prepare data source
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
        
        # Add hover tool for smaller datasets
        if len(self.data) < 50000:
            hover = HoverTool(tooltips=[
                ("Cluster", "@cluster"),
                ("Classification", "@classification")
            ])
            # Remove existing hover tools first
            self.plot.toolbar.tools = [tool for tool in self.plot.toolbar.tools if not isinstance(tool, HoverTool)]
            self.plot.add_tools(hover)
        
        # Create scatter plot
        self.scatter = self.plot.scatter(
            'x', 'y', 
            source=self.source,
            size=3,
            color='colors',
            alpha='alphas',
            selection_color="red",
            nonselection_alpha=0.1
        )

        no_color_bar = ['clus', 'session', 'edge']

        if self.view_mode in no_color_bar and self.color_bar is not None and self.color_bar in self.plot.right:
            # if in cluster mode, remove color bar if present
            self.plot.right.remove(self.color_bar)
        elif self.view_mode in no_color_bar:
            self.color_bar = None
        else:
            if self.color_bar is not None and self.color_bar in self.plot.right:
                self.plot.right.remove(self.color_bar)

            if self.view_mode == 'prob':
                self.color_bar = ColorBar(
                    color_mapper=self.prob_mapper,
                    label_standoff=12,
                    border_line_color=None,
                    location=(0, 0),
                    title="Probability"
                )

            elif self.view_mode == 'snr':
                self.color_bar = ColorBar(
                    color_mapper=LinearColorMapper(palette=Inferno256, low=np.min(self.shot_noise), high=np.max(self.shot_noise)),
                    label_standoff=12,
                    border_line_color=None,
                    location=(0, 0),
                    title="Shot Noise"
                )

            elif self.view_mode == 'size':
                self.color_bar = ColorBar(
                    color_mapper=LinearColorMapper(palette=Viridis256, low=np.min(self.footprint_size), high=np.max(self.footprint_size)),
                    label_standoff=12,
                    border_line_color=None,
                    location=(0, 0),
                    title="Footprint Size"
                )

            self.plot.add_layout(self.color_bar, 'right')
        
        # Set up tap callback for cluster mode
        if self.source:
            self.source.selected.on_change('indices', self.on_point_tap)
        
    def get_selected_indices(self):
        """Get currently selected point indices"""
        if self.source is None:
            return []
        return self.source.selected.indices
    
    def clear_selection(self):
        """Clear current selection"""
        if self.source is not None:
            self.source.selected.indices = []