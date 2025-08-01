from tqdm.keras import TqdmCallback

class CustomTqdmCallback(TqdmCallback):
    def __init__(self, *args, **kwargs):
        kwargs.update({
            'dynamic_ncols': True,  # Auto-adjust progress bar width
            'bar_format': '{l_bar}{bar:20}{r_bar}',  # Limit bar length
            'leave': False  # Cleaner output on finish
        })
        super().__init__(*args, **kwargs)
    
    def __deepcopy__(self, memo):
        # Return a fresh instance with the same configuration.
        return CustomTqdmCallback(verbose=self.verbose)
