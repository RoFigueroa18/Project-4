from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load('best_voting_regressor.pkl')

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/form')
def form():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = {
        'Brand': request.form['Brand'],
        'Model.Number': request.form['Model.Number'],
        'Fuel.Type': request.form['Fuel.Type'],
        'Body.Type': request.form['Body.Type'],
        'Gearbox.Type': request.form['Gearbox.Type'],
        'Drivetrain': request.form['Drivetrain'],
        'Engine.Type': request.form['Engine.Type'],
        'Power.hp': float(request.form['Power.hp']),
        'Seats': int(request.form['Seats']),
        'Doors': int(request.form['Doors']),
        'Torque.lbft': float(request.form['Torque.lbft']),
        'Height.in': float(request.form['Height.in']),
        'Length.in': float(request.form['Length.in']),
        'Width.in': float(request.form['Width.in']),
        'MPG.City': float(request.form['MPG.City']),
        'MPG.Highway': float(request.form['MPG.Highway']),
        'Tyre.Pressure.Monitor': request.form['Tyre.Pressure.Monitor'],
        'Cassette.Player': request.form['Cassette.Player'],
        'Heater': request.form['Heater'],
        'Electric.Adjustable.Seats': request.form['Electric.Adjustable.Seats'],
        'Central.Locking': request.form['Central.Locking'],
        'Power.Steering': request.form['Power.Steering'],
        'Halogen.Headlamps': request.form['Halogen.Headlamps'],
        'Seat.Belt.Warning': request.form['Seat.Belt.Warning'],
        'Adjustable.Seats': request.form['Adjustable.Seats'],
        'Power.Door.Locks': request.form['Power.Door.Locks'],
        'Height.Adjustable.Driving.Seat': request.form['Height.Adjustable.Driving.Seat'],
        'Vehicle.Stability.Control.System': request.form['Vehicle.Stability.Control.System'],
        'Clearance.in': float(request.form['Clearance.in']),
        'Speakers.Front...back': request.form['Speakers.Front...back'],
        'Smart.Access.Card.Entry': request.form['Smart.Access.Card.Entry'],
        'Power.Windows': request.form['Power.Windows'],
        'Fog.Lights.Front...Back': request.form['Fog.Lights.Front...Back'],
        'DVD.Player': request.form['DVD.Player'],
        'Xenon.Headlamps': request.form['Xenon.Headlamps'],
        'Centrally.Mounted.Fuel.Tank': request.form['Centrally.Mounted.Fuel.Tank'],
        'Electric.Folding.Rear.View.Mirror': request.form['Electric.Folding.Rear.View.Mirror'],
        'Rear.Reading.Lamp': request.form['Rear.Reading.Lamp'],
        'Keyless.Entry': request.form['Keyless.Entry'],
        'Adjustable.Steering.Column': request.form['Adjustable.Steering.Column'],
        'CD.Player': request.form['CD.Player'],
        'Central.Locking.1': request.form['Central.Locking.1'],
        'Smoke.Headlamps': request.form['Smoke.Headlamps'],
        'Digital.Clock': request.form['Digital.Clock'],
        'Brake.Assist': request.form['Brake.Assist'],
        'Engine.Check.Warning': request.form['Engine.Check.Warning'],
        'Rear.Camera': request.form['Rear.Camera'],
        'Automatic.Climate.Control': request.form['Automatic.Climate.Control'],
        'Night.Rear.View.Mirror': request.form['Night.Rear.View.Mirror'],
        'Anti.Theft.Device': request.form['Anti.Theft.Device'],
        'Parking.Sensors': request.form['Parking.Sensors'],
        'Touch.Screen': request.form['Touch.Screen'],
        'Outside.Temperature.Display': request.form['Outside.Temperature.Display'],
        'Cylinders': int(request.form['Cylinders']),
        'Displacement.l': float(request.form['Displacement.l']),
        'Rear.Seat.Belts': request.form['Rear.Seat.Belts'],
        'Door.Ajar.Warning': request.form['Door.Ajar.Warning'],
        'Radio': request.form['Radio'],
        'Anti.Lock.Braking': request.form['Anti.Lock.Braking'],
        'Air.Conditioner': request.form['Air.Conditioner'],
        'Removable.Convertible.Top': request.form['Removable.Convertible.Top'],
        'Audio.System.Remote.Control': request.form['Audio.System.Remote.Control'],
        'AntiLock.Braking.System': request.form['AntiLock.Braking.System'],
        'USB...Auxiliary.Input': request.form['USB...Auxiliary.Input'],
        'Engine.Immobilizer': request.form['Engine.Immobilizer'],
        'Bluetooth.Connectivity': request.form['Bluetooth.Connectivity'],
        'Wheelbase.in': float(request.form['Wheelbase.in']),
        'Crash.Sensor': request.form['Crash.Sensor'],
        'Low.Fuel.Warning.Light': request.form['Low.Fuel.Warning.Light'],
        'Voice.Control': request.form['Voice.Control'],
        'Leather.Seats': request.form['Leather.Seats'],
        'Child.Safety.Locks': request.form['Child.Safety.Locks'],
        'Bottle.Holder': request.form['Bottle.Holder']
    }

    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]

    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
