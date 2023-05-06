def prediction_plot(model_prediction, acutal_data, city_name):
    # Inputs:
        # model_prediction = sj_pred
        # acutal_data      = y_sj_test
        # city_name        = 'San Juan'
    length_sj = np.arange(len(model_prediction))

    trace0 = dict(
        type='scatter', 
        x=length_sj, 
        y=acutal_data,
        name="Actual cases")

    trace1 = dict(
        type='scatter', 
        x=length_sj, 
        y=model_prediction,
        name="Predicted cases")
    
    # Layout
    mylayout = dict(
        title=city_name,
        xaxis=dict(title='Weeks'),
        yaxis=dict(title='Total Cases'))

    # Figure
    fig = go.Figure(data=[trace0, trace1],layout=mylayout) 
    fig.show()
    
    fig.write_image("figures/"+ city_name + ".png")