class HopsworksNoAPIKey(Exception):
    """
    Raised when the API key has not been correctly configures
    """
    pass


class HopsworksLoginError(Exception):
    """
    Raised when failing to log in to Hopsworks
    """
    pass


class HopsworksGetFeatureStoreError(Exception):
    """
    Raised when failing to log in to Hopsworks
    """
    pass


class HopsworksGetFeatureGroupError(Exception):
    """
    Raised when failing to get a Feature Group from Hopsworks
    """
    pass


class HopsworksQueryReadError(Exception):
    """
    Raised when failing to make a 'read' query from Hopsworks
    """
    pass


class HopsworksFeatureGroupInsertError(Exception):
    """
    Raised when failing to insert an entry into a Feature Group on Hopsworks
    """
    pass