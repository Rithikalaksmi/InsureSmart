import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from PIL import Image, ImageTk, ImageSequence
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image, ImageTk
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import seaborn as sns

# Load dataset
def load_dataset():
    try:
        data = pd.read_csv('C:/Users/kungu/Documents/SEM 4/PA project/insurance.csv')
        return data
    except Exception as e:
        messagebox.showerror("Error", str(e))
        return None

# Function to clean the dataset
def clean_dataset(data):
    # Handle missing values
    data.dropna(inplace=True)  
    
    # Encode categorical variables
    data = pd.get_dummies(data, drop_first=True)
    
    # Remove outliers (assuming 'MEDICAL COST' is the target variable)
    z_scores = np.abs(stats.zscore(data['MEDICAL COST']))
    data = data[(z_scores < 3)]  
    return data

# Function to display regression results and inference
def display_regression_results():
    try:
        data = load_dataset()
        if data is not None:
            # Clean the dataset
            data = clean_dataset(data)
            
            # Split the dataset into independent and dependent variables
            X = data.drop('MEDICAL COST', axis=1)
            y = data['MEDICAL COST']
            
            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Initialize the linear regression model
            model = LinearRegression()
            
            # Fit the model to the training data
            model.fit(X_train, y_train)
            
            # Predict charges for the test set
            y_pred = model.predict(X_test)
            
            # Calculate the mean squared error
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            result_message = f"Mean Squared Error: {mse}\nR-squared: {r2}\nMean Absolute Error: {mae}"
            
            # Show the regression results
            result_window = tk.Toplevel()
            result_window.title("Regression Results")
            result_label = tk.Label(result_window, text=result_message)
            result_label.pack()
            
            # Add Inference button
            inference_button = ttk.Button(result_window, text="INFERENCE", command=lambda: show_regression_inference(result_window))
            inference_button.pack()
            
            # Plot the regression line
            plt.figure(figsize=(8, 6))
            plt.scatter(y_test, y_pred, color='blue', label='Actual vs. Predicted')
            plt.plot(y_test, y_test, color='red', linestyle='--', label='Ideal line')
            plt.title('Actual vs. Predicted Medical Costs')
            plt.xlabel('Actual Medical Costs')
            plt.ylabel('Predicted Medical Costs')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            # Convert the plot to a tkinter-compatible format
            plt.savefig('regression_plot.png')

            # Display the plot in the result window
            plot_image = Image.open('regression_plot.png')
            plot_photo = ImageTk.PhotoImage(plot_image)
            plot_label = tk.Label(result_window, image=plot_photo)
            plot_label.image = plot_photo
            plot_label.pack()
            
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to show regression inference
def show_regression_inference(result_window):
    inference_message = "The regression analysis indicates the following:\n\n"\
                        "1. Mean Squared Error (MSE): Measures the average squared difference between the actual and predicted values. Lower values indicate better model performance.\n"\
                        "2. R-squared (R2): Represents the proportion of variance in the dependent variable that is predictable from the independent variables. Higher values indicate better fit.\n"\
                        "3. Mean Absolute Error (MAE): Measures the average absolute difference between the actual and predicted values. Lower values indicate better model performance."

    # Display the inference message
    messagebox.showinfo("Regression Inference", inference_message)

# Function to display test dashboard
def test_dashboard(main_window):
    # Hide the main window
    main_window.withdraw()

    test_window = tk.Toplevel(main_window)
    test_window.title("Tests Dashboard")
    test_window_width = 1600
    test_window_height = 800
    test_window.geometry(f"{test_window_width}x{test_window_height}")

    # Resize background image
    bg_image = Image.open("C:/Users/kungu/Documents/SEM 4/PA project/bggif.gif")
    bg_image = bg_image.resize((test_window_width, test_window_height), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
    
    # Add resized background image to test dashboard
    bg_label = tk.Label(test_window, image=bg_photo)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)
    bg_label.image = bg_photo

    button_width = 300
    button_height = 70
    button_padding = 20
    # Create buttons for different tests
    anova_button = ttk.Button(test_window, text="ANOVA", command=perform_anova, style='My.TButton')
    anova_button.place(relx=0.5, rely=0.3, anchor='center', width=button_width, height=button_height)

    ztest_button = ttk.Button(test_window, text="Z-TEST", command=perform_ztest, style='My.TButton')
    ztest_button.place(relx=0.5, rely=0.4, anchor='center', width=button_width, height=button_height)

    corr_button = ttk.Button(test_window, text="CORRELATION ANALYSIS", command=perform_correlation, style='My.TButton')
    corr_button.place(relx=0.5, rely=0.5, anchor='center', width=button_width, height=button_height)

    # Create back button
    back_button = ttk.Button(test_window, text="<---Back", command=lambda: show_main_window(main_window, test_window), style='My.TButton')
    back_button.place(relx=0.5, rely=0.6, anchor='center', width=button_width, height=button_height)

def show_main_window(main_window, test_window):
    # Destroy the test window and show the main window
    test_window.destroy()
    main_window.deiconify()

# Function to perform ANOVA
def perform_anova():
    try:
        data = load_dataset()
        if data is not None:
            f_stat_age, p_value_age = stats.f_oneway(data['MEDICAL COST'], data['AGE'])
            f_stat_bmi, p_value_bmi = stats.f_oneway(data['MEDICAL COST'], data['BMI'])
            f_stat_dependent, p_value_dependent = stats.f_oneway(data['MEDICAL COST'], data['DEPENDENT'])
            f_stat_drug_addict, p_value_drug_addict = stats.f_oneway(data[data['DRUG ADDICT'] == 'yes']['MEDICAL COST'], data[data['DRUG ADDICT'] == 'no']['MEDICAL COST'])
            f_stat_gender, p_value_gender = stats.f_oneway(data[data['GENDER'] == 'female']['MEDICAL COST'], data[data['GENDER'] == 'male']['MEDICAL COST'])

            # Inference for ANOVA
            inference_age = "Age: There is strong evidence to suggest that age significantly affects the dependent variable.\n"\
                            "The extremely low p-value indicates that age is highly likely to have a significant impact on the dependent variable."

            inference_bmi = "BMI: Similar to age, BMI shows a highly significant effect on the dependent variable. The extremely low p-value indicates that BMI is highly likely to have a significant impact on the dependent variable."

            inference_dependent = "Dependent Variable: The results suggest that the independent variables collectively have a significant effect on the dependent variable. The extremely low p-value indicates strong evidence against the null hypothesis, suggesting that at least one of the independent variables has a significant effect on the dependent variable."

            inference_drug_addict = "Drug Addiction: Drug addiction appears to significantly affect the dependent variable. The extremely low p-value suggests that drug addiction is highly likely to have a significant impact on the dependent variable."

            inference_gender = "Gender: Gender also seems to have a significant effect on both the dependent variable and medical cost. While the p-value for the dependent variable is very low, suggesting a strong significance, the p-value for medical cost is marginal but still significant at conventional levels."

            messagebox.showinfo("ANOVA Results", 
                                f"Age - F-statistic: {f_stat_age}, p-value: {p_value_age}\n"
                                f"BMI - F-statistic: {f_stat_bmi}, p-value: {p_value_bmi}\n"
                                f"Dependent - F-statistic: {f_stat_dependent}, p-value: {p_value_dependent}\n"
                                f"Drug Addict - F-statistic: {f_stat_drug_addict}, p-value: {p_value_drug_addict}\n"
                                f"Gender - F-statistic: {f_stat_gender}, p-value: {p_value_gender}\n"
                                f"\n"
                                f"Inference:\n"
                                f"{inference_age}\n"
                                f"{inference_bmi}\n"
                                f"{inference_dependent}\n"
                                f"{inference_drug_addict}\n"
                                f"{inference_gender}\n")
    except Exception as e:
        messagebox.showerror("Error", str(e))
# Function to perform Z-test
def perform_ztest():
    try:
        data = load_dataset()
        if data is not None:
            female_count = data[data['GENDER'] == 'female'].shape[0]
            male_count = data[data['GENDER'] == 'male'].shape[0]
            total_count = female_count + male_count
            
            z_stat, p_value = proportions_ztest([female_count, male_count], [total_count, total_count])

            inference = "The Z-test results indicate the following:\n\n"\
                        "The null hypothesis assumes that there is no significant difference in the proportions of female and male individuals.\n"\
                        "A p-value less than 0.05 would suggest rejecting the null hypothesis, indicating a significant difference.\n"\
                        "In this case, the extremely low p-value indicates that there is a significant difference between the proportions of females and males."

            messagebox.showinfo("Z-Test Results", f"Z-statistic: {z_stat}, p-value: {p_value}\n\nInference:\n{inference}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to perform correlation analysis
def perform_correlation():
    try:
        data = load_dataset()
        if data is not None:
            # Select relevant columns for correlation analysis
            correlation_columns = ['AGE', 'BMI', 'DEPENDENT', 'MEDICAL COST']
            correlation_matrix = data[correlation_columns].corr()

            # Inferences for correlation
            inference_messages = [
                "Age and BMI: There is a weak positive correlation between age and BMI (0.109), indicating that as age increases, BMI tends to increase slightly.",
                "Age and Dependent Variable: There is a very weak positive correlation between age and the dependent variable (0.042), suggesting a minimal association between age and the dependent variable.",
                "Age and Medical Cost: There is a moderate positive correlation between age and medical cost (0.299), indicating that as age increases, medical costs tend to increase moderately.",
                "BMI and Dependent Variable: There is a very weak positive correlation between BMI and the dependent variable (0.012), suggesting a minimal association between BMI and the dependent variable.",
                "BMI and Medical Cost: There is a weak positive correlation between BMI and medical cost (0.198), indicating that as BMI increases, medical costs tend to increase slightly.",
                "Dependent Variable and Medical Cost: There is a very weak positive correlation between the dependent variable and medical cost (0.068), suggesting a minimal association between the two."
            ]

            # Create a message with correlation matrix and inferences
            correlation_message = "Correlation Matrix:\n{}\n\n".format(correlation_matrix)
            inference_message = "\n\n".join(inference_messages)
            messagebox.showinfo("Correlation Matrix", correlation_message + inference_message)
    except Exception as e:
        messagebox.showerror("Error", str(e))


# Function to perform PCA
def perform_pca():
    try:
        data = load_dataset()
        if data is not None:
            data = clean_dataset(data)
            X = data.drop('MEDICAL COST', axis=1)
            y = data['MEDICAL COST']
            X_scaled = StandardScaler().fit_transform(X)
            pca = PCA(n_components=2)
            pca_components = pca.fit_transform(X_scaled)
            pca_df = pd.DataFrame(data=pca_components, columns=['Principal Component 1', 'Principal Component 2'])
            pca_df = pd.concat([pca_df, y.reset_index(drop=True)], axis=1)
            
            # Compute explained variance ratio
            explained_variance = pca.explained_variance_ratio_
            
            # Create a new window for displaying PCA results
            pca_window = tk.Toplevel()
            pca_window.title("PCA Results")
            
            # Create a frame to contain both the numerical output and the plot
            frame = ttk.Frame(pca_window)
            frame.pack(padx=10, pady=10)
            
            # Display numerical output
            pca_output_text = "PCA Results:\n\n"
            pca_output_text += f"Explained Variance Ratio:\nComponent 1: {explained_variance[0]:.2f}\nComponent 2: {explained_variance[1]:.2f}\n"
            pca_output_label = tk.Label(frame, text=pca_output_text)
            pca_output_label.grid(row=0, column=0, padx=10, pady=10)
            
            # Plot PCA
            pca_plot_frame = ttk.Frame(frame)
            pca_plot_frame.grid(row=0, column=1, padx=10, pady=10)
            plt.figure(figsize=(8, 6))
            plt.scatter(pca_df['Principal Component 1'], pca_df['Principal Component 2'], c=pca_df['MEDICAL COST'], cmap='viridis')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.title('PCA - Medical Cost')
            plt.colorbar(label='Medical Cost')
            plt.tight_layout()
            plt.savefig('pca_plot.png')
            pca_plot_image = Image.open('pca_plot.png')
            pca_plot_photo = ImageTk.PhotoImage(pca_plot_image)
            pca_plot_label = tk.Label(pca_plot_frame, image=pca_plot_photo)
            pca_plot_label.image = pca_plot_photo
            pca_plot_label.pack()

            # Add Inference button
            inference_button = ttk.Button(pca_window, text="INFERENCE", command=lambda: show_pca_inference(pca_window))
            inference_button.pack()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to show PCA inference
def show_pca_inference(pca_window):
    inference_message = "The PCA analysis indicates the following:\n\n"\
                        "1. Principal Component 1 and Principal Component 2 capture the majority of the variance in the dataset.\n"\
                        "2. Data points are spread across the principal components, indicating variability in medical costs across different dimensions.\n"\
                        "3. The color gradient represents the medical costs, showing how different components correlate with medical cost levels."

    # Display the inference message
    messagebox.showinfo("PCA Inference", inference_message)


def perform_factor_analysis():
    try:
        data = load_dataset()
        if data is not None:
            data = clean_dataset(data)
            X = data.drop('MEDICAL COST', axis=1)
            fa = FactorAnalysis(n_components=2)
            fa_components = fa.fit_transform(X)
            fa_df = pd.DataFrame(data=fa_components, columns=['Factor 1', 'Factor 2'])
            fa_df = pd.concat([fa_df, data['MEDICAL COST'].reset_index(drop=True)], axis=1)
            
            # Create a new window for displaying Factor Analysis results
            fa_window = tk.Toplevel()
            fa_window.title("Factor Analysis Results")
            
            # Create a frame to contain both the numerical output and the plot
            frame = ttk.Frame(fa_window)
            frame.pack(padx=10, pady=10)
            
            # Display numerical output
            fa_output_text = "Factor Analysis Results:\n\n"
            fa_output_text += "Factor Loadings:\n" + str(fa.components_) + "\n"
            fa_output_label = tk.Label(frame, text=fa_output_text)
            fa_output_label.grid(row=0, column=0, padx=10, pady=10)
            
            # Plot Factor Analysis
            fa_plot_frame = ttk.Frame(frame)
            fa_plot_frame.grid(row=0, column=1, padx=10, pady=10)
            plt.figure(figsize=(8, 6))
            plt.scatter(fa_df['Factor 1'], fa_df['Factor 2'], c=fa_df['MEDICAL COST'], cmap='viridis')
            plt.xlabel('Factor 1')
            plt.ylabel('Factor 2')
            plt.title('Factor Analysis - Medical Cost')
            plt.colorbar(label='Medical Cost')
            plt.tight_layout()
            plt.savefig('factor_analysis_plot.png')
            fa_plot_image = Image.open('factor_analysis_plot.png')
            fa_plot_photo = ImageTk.PhotoImage(fa_plot_image)
            fa_plot_label = tk.Label(fa_plot_frame, image=fa_plot_photo)
            fa_plot_label.image = fa_plot_photo
            fa_plot_label.pack()

            # Add Inference button
            inference_button = ttk.Button(fa_window, text="INFERENCE", command=lambda: show_fa_inference(fa_window))
            inference_button.pack()
    except Exception as e:
        messagebox.showerror("Error", str(e))


# Function to show Factor Analysis inference
def show_fa_inference(fa_window):
    inference_message = "The Factor Analysis indicates the following:\n\n"\
                        "1. Factor 1 and Factor 2 capture the underlying structure in the dataset, explaining the correlations between variables.\n"\
                        "2. The scatter plot shows how data points are distributed across the identified factors.\n"\
                        "3. The color gradient represents the medical costs, illustrating how different factors relate to medical cost levels."

    # Display the inference message
    messagebox.showinfo("Factor Analysis Inference", inference_message)

# Function to perform hierarchical clustering
def perform_hierarchical_clustering():
    try:
        data = load_dataset()
        if data is not None:
            data = clean_dataset(data)
            X = data.drop('MEDICAL COST', axis=1)
            X_scaled = StandardScaler().fit_transform(X)
            linked = linkage(X_scaled, 'ward')

            # Create a new window for displaying clustering results
            clustering_window = tk.Toplevel()
            clustering_window.title("Hierarchical Clustering Results")

            # Create a frame to contain both the dendrogram plot and numerical output
            frame = ttk.Frame(clustering_window)
            frame.pack(padx=10, pady=10)

            # Display the dendrogram plot
            dendrogram_frame = ttk.Frame(frame)
            dendrogram_frame.grid(row=0, column=0, padx=10, pady=10)
            dendrogram_label = tk.Label(dendrogram_frame, text="Hierarchical Clustering Dendrogram")
            dendrogram_label.pack()
            plt.figure(figsize=(8, 6))
            dendrogram(linked, orientation='top')
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Data Points')
            plt.ylabel('Distance')
            plt.tight_layout()
            plt.savefig('dendrogram_plot.png')
            dendrogram_image = Image.open('dendrogram_plot.png')
            dendrogram_photo = ImageTk.PhotoImage(dendrogram_image)
            dendrogram_label = tk.Label(dendrogram_frame, image=dendrogram_photo)
            dendrogram_label.image = dendrogram_photo
            dendrogram_label.pack()

            # Compute and display cluster statistics
            cluster_stats_frame = ttk.Frame(frame)
            cluster_stats_frame.grid(row=0, column=1, padx=10, pady=10)
            clusters = fcluster(linked, 2, criterion='maxclust')  # Adjust the number of clusters as needed
            cluster_counts = np.bincount(clusters)
            cluster_stats_text = "Cluster Statistics:\n\n"
            for i, count in enumerate(cluster_counts):
                cluster_stats_text += f"Cluster {i+1}: {count} data points\n"
            cluster_stats_label = tk.Label(cluster_stats_frame, text=cluster_stats_text)
            cluster_stats_label.pack()

            # Add Inference button
            inference_button = ttk.Button(clustering_window, text="INFERENCE", command=lambda: show_dendrogram_inference(clustering_window))
            inference_button.pack()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to show hierarchical clustering inference
def show_dendrogram_inference(dendrogram_window):
    inference_message = "The hierarchical clustering dendrogram indicates the following:\n\n"\
                        "1. Data points are grouped based on their similarity in features related to medical costs.\n"\
                        "2. The vertical axis represents the distance or dissimilarity between clusters.\n"\
                        "3. The dendrogram can help identify potential clusters or patterns in the dataset."

    # Display the inference message
    messagebox.showinfo("Hierarchical Clustering Inference", inference_message)

# Function to perform KMeans clustering
def perform_kmeans_clustering():
    try:
        data = load_dataset()
        if data is not None:
            # Clean the dataset
            data = clean_dataset(data)
            
            # Standardize the features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Perform KMeans clustering
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # Explicitly set n_init to suppress the warning
            cluster_labels = kmeans.fit_predict(scaled_data)
            
            # Add cluster labels to the dataset
            data['Cluster'] = cluster_labels
            
            # Display the clustering results
            cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)  # Convert cluster centers back to original scale
            
            # Plot the clusters along with centroids
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.scatterplot(data=data, x='AGE', y='MEDICAL COST', hue='Cluster', palette='viridis', ax=ax)
            ax.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker='X', color='red', s=100, label='Centroids')
            ax.set_title('KMeans Clustering')
            ax.set_xlabel('Age')
            ax.set_ylabel('Medical Cost')
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            
            # Add inference button
            inference_button = tk.Button(fig.canvas.manager.window, text="INFERENCE", command=lambda: show_kmeans_inference(fig.canvas.manager.window))
            inference_button.place(relx=0.85, rely=0.05)
            
            # Add numerical output below the plot
            numerical_output = tk.Text(fig.canvas.manager.window, height=4, width=50)
            numerical_output.place(relx=0.5, rely=0.9, anchor='center')
            numerical_output.insert(tk.END, f"Cluster Centers:\n{cluster_centers}")
            
            plt.show()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to show KMeans clustering inference
def show_kmeans_inference(kmeans_window):
    inference_message = "The KMeans clustering indicates the following:\n\n"\
                        "1. Data points are grouped into clusters based on their similarity in features.\n"\
                        "2. Centroids (marked in red) represent the centers of the clusters.\n"\
                        "3. The scatter plot visualizes the distribution of data points and cluster centroids."

    # Display the inference message
    messagebox.showinfo("KMeans Clustering Inference", inference_message)



# Main function
def main():
    main_window = tk.Tk()
    main_window.title("Medical Insurance Data Analysis")
    main_window_width = 1600
    main_window_height = 800
    main_window.geometry(f"{main_window_width}x{main_window_height}")

    # Load and set the background image
    bg_image = Image.open("C:/Users/kungu/Documents/SEM 4/PA project/bggif.gif")
    bg_image = bg_image.resize((main_window_width, main_window_height), Image.LANCZOS)  # Use LANCZOS method for resizing
    frames = [ImageTk.PhotoImage(frame.copy()) for frame in ImageSequence.Iterator(bg_image)]
    
    bg_label = tk.Label(main_window)
    bg_label.place(x=0, y=0, relwidth=1, relheight=1)

    def update_frame(index):
        frame = frames[index]
        bg_label.configure(image=frame)
        main_window.after(100, update_frame, (index + 1) % len(frames))

    update_frame(0)

    # Create buttons
    button_width = 300
    button_height = 70
    button_padding = 20

    regression_button = ttk.Button(main_window, text="Perform Regression Analysis", command=display_regression_results, style='My.TButton')
    regression_button.place(relx=0.5, rely=0.3, anchor="center", width=button_width, height=button_height)

    dashboard_button = ttk.Button(main_window, text="Tests Dashboard", command=lambda: test_dashboard(main_window), style='My.TButton')
    dashboard_button.place(relx=0.5, rely=0.4, anchor="center", width=button_width, height=button_height)

    pca_button = ttk.Button(main_window, text="PCA", command=perform_pca, style='My.TButton')
    pca_button.place(relx=0.5, rely=0.5, anchor="center", width=button_width, height=button_height)

    factor_analysis_button = ttk.Button(main_window, text="Factor Analysis", command=perform_factor_analysis, style='My.TButton')
    factor_analysis_button.place(relx=0.5, rely=0.6, anchor="center", width=button_width, height=button_height)

    clustering_button = ttk.Button(main_window, text="Hierarchical Clustering", command=perform_hierarchical_clustering, style='My.TButton')
    clustering_button.place(relx=0.5, rely=0.7, anchor="center", width=button_width, height=button_height)

    kmeans_button = ttk.Button(main_window, text="KMeans Clustering", command=perform_kmeans_clustering, style='My.TButton')
    kmeans_button.place(relx=0.5, rely=0.8, anchor="center", width=button_width, height=button_height)

    main_window.mainloop()
    
if __name__ == "__main__":
    main()
