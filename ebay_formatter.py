# ebay_formatter.py
import pandas as pd
from datetime import datetime, timezone

DEFAULTS = {
    "category_id": 262042,
    "category_name": "/Collectibles/Postcards & Supplies/Postcards/Topographical Postcards",
    "start_price": 7.99,
    "quantity": 1,
    "condition_id": "3000-Used",
    "listing_format": "FixedPrice",
    "duration": "GTC",
    "location": "Greensboro NC",
    "shipping_service": "USPSFirstClass",
    "shipping_cost": 0,
    "handling_time": 3,                 # the mysterious “3” column 😄
    "shipping_profile": "Standard",
    "return_profile": "No Returns Accepted",
    "payment_profile": "eBay Managed Payments",
    "licensed_reprint": "Original",
    "postage_condition": "Posted",
}

def reformat_for_ebay(raw,         # DataFrame *or* list-of-dicts you already have
                      schedule_time: str | None = None,
                      **overrides):
    """
    Turn the output of `save_postcards_to_csv` into the 28-column eBay template.

    Parameters
    ----------
    raw : pandas.DataFrame | list[dict]
        The data your current code emits (columns: SKU, front_image_link, … Era).
    schedule_time : str | None
        ISO-8601 UTC time when you want the listing scheduled.
        If None, the current UTC time is used.
    **overrides
        Any constant you want different from DEFAULTS can be passed as a kwarg
        (e.g. start_price=9.95).

    Returns
    -------
    pandas.DataFrame
        Ready to be written with `.to_csv("ebay_ready.csv", index=False)`.
    """

    # Make sure we’re working with a DataFrame
    df = pd.DataFrame(raw).copy()

    cfg = {**DEFAULTS, **overrides}
    if schedule_time is None:
        # eBay likes millisecond precision with a trailing Z
        schedule_time = datetime.now(timezone.utc)\
                                .isoformat(timespec="milliseconds")\
                                .replace("+00:00", "Z")

    # Build the File-Exchange shell
    ebay_df = pd.DataFrame({
        "Custom Label (SKU)":             df["SKU"],
        "Category ID":                    cfg["category_id"],
        "Category Name":                  cfg["category_name"],
        "Title":                          df["Title"],
        "Schedule Time":                  schedule_time,
        "Start price":                    cfg["start_price"],
        "Quantity":                       cfg["quantity"],
        "Item photo URL":                 "#N/A",               # eBay placeholder
        "URL1":                           df["front_image_link"],
        "Pipe":                           "|",
        "URL2":                           df["back_image_link"].fillna(""),
        "Condition ID":                   cfg["condition_id"],
        "Description":                    df["Description"],
        "Format":                         cfg["listing_format"],
        "Duration":                       cfg["duration"],
        "Location":                       cfg["location"],
        "Shipping service 1 option":      cfg["shipping_service"],
        "Shipping service 1 cost":        cfg["shipping_cost"],
        "https://www.ebay.com/sh/reports/uploads": cfg["handling_time"],
        "Shipping profile name":          cfg["shipping_profile"],
        "Return profile name":            cfg["return_profile"],
        "Payment profile name":           cfg["payment_profile"],
        "C:Region":                       df["Region"],
        "C:Country":                      df["Country"],
        "C:City":                         df["City"],
        "C:Era":                          df["Era"],
        "C:Original/Licensed Reprint":    cfg["licensed_reprint"],
        "C:Postage Condition":            cfg["postage_condition"],
    })

    # Absolute column order matters to eBay
    ordered_cols = [
        "Custom Label (SKU)", "Category ID", "Category Name", "Title",
        "Schedule Time", "Start price", "Quantity", "Item photo URL",
        "URL1", "Pipe", "URL2", "Condition ID", "Description", "Format",
        "Duration", "Location", "Shipping service 1 option",
        "Shipping service 1 cost", "https://www.ebay.com/sh/reports/uploads",
        "Shipping profile name", "Return profile name", "Payment profile name",
        "C:Region", "C:Country", "C:City", "C:Era",
        "C:Original/Licensed Reprint", "C:Postage Condition"
    ]
    return ebay_df[ordered_cols]
