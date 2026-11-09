
import pandas as pd
import pickle

# Import ML tools
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


try:

    # 1. Load dataset
    df = pd.read_csv('data.csv')

    # 2. Remove unnecessary columns
    df = df.drop(['date', 'street'], axis=1)

    # 3. City mapping
    city_mapping = {
        'Shoreline': 0, 'Seattle': 1, 'Kent': 2, 'Bellevue': 3,
        'Redmond': 4, 'Maple Valley': 5, 'North Bend': 6,
        'Lake Forest Park': 7, 'Sammamish': 8, 'Auburn': 9,
        'Des Moines': 10, 'Bothell': 11, 'Federal Way': 12,
        'Kirkland': 13, 'Issaquah': 14, 'Woodinville': 15,
        'Normandy Park': 16, 'Fall City': 17, 'Renton': 18,
        'Carnation': 19, 'Snoqualmie': 20, 'Duvall': 21,
        'Burien': 22, 'Covington': 23, 'Inglewood-Finn Hill': 24,
        'Kenmore': 25, 'Newcastle': 26, 'Mercer Island': 27,
        'Black Diamond': 28, 'Ravensdale': 29, 'Clyde Hill': 30,
        'Algona': 31, 'Skykomish': 32, 'Tukwila': 33,
        'Vashon': 34, 'Yarrow Point': 35, 'SeaTac': 36,
        'Medina': 37, 'Enumclaw': 38, 'Snoqualmie Pass': 39,
        'Pacific': 40, 'Beaux Arts Village': 41, 'Preston': 42,
        'Milton': 43
    }

    # 4. ZIP code mapping
    zip_mapping = {
        'WA 98133': 0, 'WA 98119': 1, 'WA 98042': 2,
        'WA 98008': 3, 'WA 98052': 4, 'WA 98115': 5,
        'WA 98038': 6, 'WA 98045': 7, 'WA 98155': 8,
        'WA 98105': 9, 'WA 98074': 10, 'WA 98106': 11,
        'WA 98007': 12, 'WA 98092': 13, 'WA 98198': 14,
        'WA 98006': 15, 'WA 98102': 16, 'WA 98011': 17,
        'WA 98125': 18, 'WA 98003': 19, 'WA 98136': 20,
        'WA 98033': 21, 'WA 98029': 22, 'WA 98117': 23,
        'WA 98034': 24, 'WA 98072': 25, 'WA 98023': 26,
        'WA 98107': 27, 'WA 98166': 28, 'WA 98116': 29,
        'WA 98024': 30, 'WA 98055': 31, 'WA 98077': 32,
        'WA 98027': 33, 'WA 98059': 34, 'WA 98075': 35,
        'WA 98014': 36, 'WA 98065': 37, 'WA 98199': 38,
        'WA 98053': 39, 'WA 98058': 40, 'WA 98122': 41,
        'WA 98103': 42, 'WA 98112': 43, 'WA 98005': 44,
        'WA 98118': 45, 'WA 98177': 46, 'WA 98004': 47,
        'WA 98019': 48, 'WA 98144': 49, 'WA 98168': 50,
        'WA 98001': 51, 'WA 98056': 52, 'WA 98146': 53,
        'WA 98028': 54, 'WA 98148': 55, 'WA 98057': 56,
        'WA 98040': 57, 'WA 98010': 58, 'WA 98051': 59,
        'WA 98031': 60, 'WA 98109': 61, 'WA 98030': 62,
        'WA 98126': 63, 'WA 98032': 64, 'WA 98178': 65,
        'WA 98288': 66, 'WA 98108': 67, 'WA 98070': 68,
        'WA 98188': 69, 'WA 98002': 70, 'WA 98039': 71,
        'WA 98022': 72, 'WA 98068': 73, 'WA 98047': 74,
        'WA 98050': 75, 'WA 98354': 76
    }

    # 5. Check for missing city and ZIP mappings
    missing_cities = set(df['city'].dropna().unique()) - set(city_mapping.keys())
    missing_zips = set(df['statezip'].dropna().unique()) - set(zip_mapping.keys())


    if missing_cities or missing_zips:
        raise ValueError(
            f"Missing city mappings: {missing_cities}\n"
            f"Missing ZIP mappings: {missing_zips}"
        )

    # 6. Convert city, ZIP code and country into numbers
    df['statezip'] = df['statezip'].map(zip_mapping).astype(int)
    df['city'] = df['city'].map(city_mapping).astype(int)
    df['country'] = df['country'].map({'USA': 0}).astype(int)

    # 7. Create price categories
    df['price_category'] = pd.cut(
        df['price'],
        bins=[0, 400000, 700000, float('inf')],
        labels=['Low', 'Medium', 'High']
    )

    # Remove rows without a price category
    df = df.dropna(subset=['price_category'])

    # 8. Separate input (X) and output (y)
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # 9. Split dataset into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 10. Create Random Forest model
    reg = RandomForestClassifier(
        criterion='entropy',
        n_estimators=100,
        random_state=42
    )

    # 11. Train the model
    reg.fit(X_train, y_train)

    # 12. Predict test data
    pred = reg.predict(X_test)

    # 13. Evaluate model performance
    accuracy = accuracy_score(y_test, pred)

    conf_matrix = confusion_matrix(
        y_test,
        pred,
        labels=['Low', 'Medium', 'High']
    )

    report = classification_report(
        y_test,
        pred,
        output_dict=True,
        zero_division=0
    )

    print("\nModel Accuracy:", round(accuracy * 100, 2), "%")

    print("\nConfusion Matrix:\n", conf_matrix)

    print("\nClassification Report:\n")
    print(classification_report(y_test, pred, zero_division=0))

    # 14. Store everything in one dictionary
    model_data = {
        'model': reg,
        'city_mapping': city_mapping,
        'zip_mapping': zip_mapping,
        'columns': list(X.columns),
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': report
    }

    # 15. Save everything in one pickle file
    with open('model.pkl', 'wb') as file:
        pickle.dump(model_data, file)

    print("\nSuccess! model.pkl created successfully.")

except FileNotFoundError:
    print("Error: data.csv was not found. Check your project folder.")

except ValueError as e:
    print("\nData or mapping error:")
    print(e)

except Exception as e:
    print("\nSomething went wrong:")
    print(e)
