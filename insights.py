import pandas as pd
import streamlit as st
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
import time
import pydeck as pdk
import numpy as np


def load_dataframe():
    return pd.read_pickle("receipts.pkl").sort_values("datetime")


df = load_dataframe()


@st.fragment
def montly_spend(df):
    with st.container():
        unique_months = df["datetime"].dt.month_name().unique()
        selected_month = st.selectbox(
            "Select Month", unique_months, index=len(unique_months) - 1
        )
    df = df[df["month"] == selected_month]
    row1 = st.columns(3)
    row1[0].metric(
        "Total Spend",
        round(df["totalPrice"].sum(), 2),
    )
    row1[1].metric(
        "Average Spend per Day",
        round(
            df.groupby("date")["totalPrice"].mean().mean(),
            2,
        ),
    )
    row1[2].metric(
        "Average Spend per Transaction",
        round(
            df["totalPrice"].mean(),
            2,
        ),
    )
    st.metric("Total Transactions", df["totalPrice"].count())

    total_spend_per_day = df.groupby("date")["totalPrice"].sum()
    fig = px.bar(
        total_spend_per_day,
        title=f"Total Spend per Day {selected_month}",
        color=total_spend_per_day,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

    row3 = st.columns(2)

    average_spend_per_transaction_time_of_day = df.groupby("time_of_day")[
        "totalPrice"
    ].mean()

    fig = px.bar(
        average_spend_per_transaction_time_of_day,
        title=f"Average Transaction value by Time of Day {selected_month}",
        labels={"index": "Day Type", "value": "Average Spend"},
        color=average_spend_per_transaction_time_of_day,
    )
    fig.update_layout(showlegend=False)
    row3[0].plotly_chart(fig)
    # make horizontal
    # Calculate the average spend per transaction for weekends and weekdays
    average_spend_per_transaction_weekend = df.groupby("weekend")["totalPrice"].mean()

    # Map the weekend/weekday codes to labels
    average_spend_per_transaction_weekend.index = (
        average_spend_per_transaction_weekend.index.map({0: "Weekday", 1: "Weekend"})
    )

    # Plot with Plotly Express
    fig = px.bar(
        average_spend_per_transaction_weekend,
        title=f"Average Spend per Transaction Weekend vs Weekday {selected_month}",
        labels={"index": "Day Type", "value": "Average Spend"},
        color=average_spend_per_transaction_weekend,
    )

    # Hide the legend
    fig.update_layout(showlegend=False)
    # Display the plot in the specified container
    row3[1].plotly_chart(fig)

    average_spend_per_transaction_week_of_month = df.groupby("week_of_month")[
        "totalPrice"
    ].mean()

    fig = px.bar(
        average_spend_per_transaction_week_of_month,
        title=f"Average Spend per Transaction by Week of Month {selected_month}",
        labels={"index": "Week of Month", "value": "Average Spend"},
        color=average_spend_per_transaction_week_of_month,
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

    business_counts = df["businessName"].value_counts()
    # most shopped at businesses horizontal bar chart
    fig = px.bar(
        business_counts,
        title=f"Most shopped at businesses {selected_month}",
        orientation="h",
        color=business_counts,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

    # add the count of transactions per postal code and the lat and long
    average_spend_per_postal_code = df.groupby("businessPostalCode").agg(
        {
            "totalPrice": "mean",
            "businessPostalCode": "count",
            "latitude": "mean",
            "longitude": "mean",
        }
    )
    # change businessPostalCode to transactionCount totalPrice to averageTransactionAmount
    average_spend_per_postal_code = average_spend_per_postal_code.rename(
        columns={
            "businessPostalCode": "transactionCount",
            "totalPrice": "averageTransactionAmount",
        }
    )
    # drop where lat long nan

    average_spend_per_postal_code = average_spend_per_postal_code.dropna(
        subset=["latitude", "longitude"]
    )

    st.subheader("Average Transaction Amount by Postal Code")
    fig = px.bar(
        average_spend_per_postal_code["averageTransactionAmount"],
        # title=f"Average Transaction Amount by Postal Code {selected_month}",
        orientation="h",
        color=average_spend_per_postal_code["transactionCount"],
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig)

    st.pydeck_chart(
        pdk.Deck(
            # map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state=pdk.ViewState(
                latitude=average_spend_per_postal_code[
                    "latitude"
                ].mean(),  # Center around mean latitude
                longitude=average_spend_per_postal_code[
                    "longitude"
                ].mean(),  # Center around mean longitude
                zoom=10,
                pitch=60,
            ),
            description="Average Transaction Amount by Postal Code",
            layers=[
                # HexagonLayer to show average transaction amount as extruded bars
                pdk.Layer(
                    "HexagonLayer",
                    data=average_spend_per_postal_code,
                    get_position=["longitude", "latitude"],
                    radius=100,  # Adjust radius to control the size of each hexagon
                    elevation_scale=4,  # Adjust scale for visible extrusion
                    extruded=True,
                    # pickable=True,
                    get_fill_color="averageTransactionAmount",
                    get_elevation_weight="averageTransactionAmount",  # Use average transaction amount for extrusion
                    # Use mean to aggregate the average transaction amounts per hexagon
                ),
            ],
        )
    )


# columns
# Index(['date', 'time', 'items', 'prices', 'paymentMethod', 'cardNumber',
#        'cardType', 'quantities', 'totalItems', 'totalPrice', 'businessName',
#        'businessAddress', 'businessPostalCode', 'latitude', 'longitude',
#        'image_path', 'datetime', 'day_of_week', 'weekend', 'day_of_month',
#        'week_of_month', 'time_of_day', 'device'],
#       dtype='object')

# spend per day

# plotly figure of spend per day


montly_spend(df)


style_metric_cards(
    background_color="#0000000",
    border_left_color="blue",
)
