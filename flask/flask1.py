
import joblib 
model = joblib.load("maram.pkl")

# تجربة تنبؤ عشوائي
sample_input = [[0.8, 11.7, 7.2, 2.3]] 
prediction = model.predict(sample_input)

print("التنبؤ:", prediction)
