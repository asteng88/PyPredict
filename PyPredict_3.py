# @version: 1.0 (2024-07-23)
# IMPORTANT WHEN SETTING UP THE .venv:
# When setting up the virtual environment, make sure to install the following packages:
# Create the environment using Python 3.10.9
# pip install pandas statsmodels openpyxl pmdarima sklearn reportlab matplotlib
# uninstall numpy and install version 1.23.5 (pip install numpy==1.23.5) - This is to ensure compatibility with the pmdarima package
# icon generated from <a href="https://www.flaticon.com/free-icons/statistics" title="statistics icons">Statistics icons created by logisstudio - Flaticon</a>

import tkinter as tk
from tkinter import filedialog
import pandas as pd
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from pmdarima.arima import auto_arima
from sklearn.linear_model import LinearRegression
import warnings
import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.lib.utils import ImageReader
import io

def select_file():
    global df, excel_file_path
    filetypes = [("Excel files", "*.xlsx *.xls")]
    filename = filedialog.askopenfilename(title="Open Excel File", filetypes=filetypes)
    if filename:
        excel_file_path = filename
        try:
            df = pd.read_excel(filename, header=None, engine='openpyxl')
            refresh_window()  # Automatically refresh the data when the file is loaded successfully
            messagebox.showinfo("Success", "Excel file loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load Excel file:\n{e}")

def refresh_window():
    global df, data_frame
    if excel_file_path == "":
        messagebox.showwarning("Warning", "Please select an Excel file first.")
        return
    try:
        # Clear previous data in data_frame
        for widget in data_frame.winfo_children():
            widget.destroy()
        
        # Get inputs
        row_headers_input = row_headers_var.get()
        col_headers_input = col_headers_var.get()
        data_end_row_input = data_end_row_var.get()
        data_end_col_input = data_end_col_var.get()
        data_orientation = data_orientation_var.get().strip().capitalize()
        data_position_input = data_position_var.get()
        
        # Convert inputs to integers if possible
        row_headers = int(row_headers_input) - 1 if row_headers_input.strip() != "" else None
        col_headers = int(col_headers_input) - 1 if col_headers_input.strip() != "" else None
        data_end_row = int(data_end_row_input) - 1 if data_end_row_input.strip() != "" else None
        data_end_col = int(data_end_col_input) - 1 if data_end_col_input.strip() != "" else None
        data_position = int(data_position_input) - 1 if data_position_input.strip() != "" else None
        
        # Debugging prints for data_orientation and data_position
        #print(f"data_orientation: '{data_orientation}'")
        #print(f"data_position: {data_position}")
        
        # Extract data
        df_display = df.copy()
        if data_end_row is not None:
            df_display = df_display.iloc[:data_end_row+1, :]
        if data_end_col is not None:
            df_display = df_display.iloc[:, :data_end_col+1]
        
        rows, cols = df_display.shape
        
        # Add padding to the left side of the data frame
        padding_frame = tk.Frame(data_frame, width=20)
        padding_frame.grid(row=0, column=0, sticky='ns')
        
        for r in range(rows):
            for c in range(cols):
                value = df_display.iat[r, c]
                e = tk.Entry(data_frame, width=10)
                e.grid(row=r, column=c+1)  # Shift columns to the right by 1 due to padding
                e.insert(tk.END, str(value))
                # Determine cell background color
                if (row_headers is not None and r == row_headers) or (col_headers is not None and c == col_headers):
                    e.configure(bg='yellow')
                elif data_position is not None and (
                    (data_orientation.lower() == 'rows' and r == data_position) or 
                    (data_orientation.lower() == 'columns' and c == data_position)):
                    #print(f"Highlighting cell at row {r}, column {c} in green") # Use for Debugging data selection
                    e.configure(bg='lightgreen')
                else:
                    e.configure(bg='white')
        
        # Update scroll region
        data_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))
        
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred:\n{e}")

def analyze_data():
    global df, analysis_result_text, analysis_fig, fig, analysis_text, data_series_used
    if excel_file_path == "":
        messagebox.showwarning("Warning", "Please select an Excel file first.")
        return
    try:
        # Get inputs
        data_orientation = data_orientation_var.get()
        data_position_input = data_position_var.get()
        num_predictions_input = num_predictions_var.get()
        selected_model = model_var.get()
        row_headers_input = row_headers_var.get()
        
        # Validate inputs
        if data_position_input.strip() == "":
            messagebox.showerror("Error", "Please specify the row or column number containing the data to analyze.")
            return
        if num_predictions_input.strip() == "":
            messagebox.showerror("Error", "Please specify the number of predictions to make.")
            return
        if row_headers_input.strip() == "":
            messagebox.showerror("Error", "Please specify the row number for headers.")
            return
        
        # Convert inputs
        data_position = int(data_position_input) - 1
        num_predictions = int(num_predictions_input)
        row_headers = int(row_headers_input) - 1
        
        # Check that data_position and num_predictions are valid
        if data_position < 0:
            messagebox.showerror("Error", "Row or column number must be a positive integer.")
            return
        if num_predictions <= 0:
            messagebox.showerror("Error", "Number of predictions must be a positive integer.")
            return
        
        # Extract the data to analyze
        if data_orientation == "Rows":
            if data_position >= len(df):
                messagebox.showerror("Error", f"The specified row number {data_position+1} exceeds the number of rows in the data.")
                return
            data_series = df.iloc[data_position, :].dropna()
            y_label = f"{df.iloc[data_position, 0]} (Row {data_position+1})"
            x_label = f"{df.iloc[row_headers, 0]}"  # Changed to use the selected row headers
        elif data_orientation == "Columns":
            if data_position >= len(df.columns):
                messagebox.showerror("Error", f"The specified column number {data_position+1} exceeds the number of columns in the data.")
                return
            data_series = df.iloc[:, data_position].dropna()
            y_label = f"{df.iloc[0, data_position]} (Column {data_position+1})"
            x_label = f"{df.iloc[row_headers, 0]}"  # Changed to use the selected row headers
        else:
            messagebox.showerror("Error", "Please select data orientation (Rows or Columns).")
            return
        
        # refresh the data frame
        refresh_window()
        
        # Store the data series used for report generation
        data_series_used = data_series.copy()
        
        # Convert data to numeric
        data_series = pd.to_numeric(data_series, errors='coerce').dropna()
        if data_series.empty:
            messagebox.showerror("Error", "No numeric data found in the specified row/column.")
            return
        
        y = data_series.values
        x = np.arange(len(y))
        
        # Check if there are enough data points
        if len(y) < 10:
            messagebox.showerror("Error", "Not enough data points for analysis (minimum 10 required).")
            return
        
        analysis_text = ""
        
        if selected_model == "ARIMA":
            # Fit ARIMA model using auto_arima
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=ConvergenceWarning)
                    model_fit = auto_arima(
                        y,
                        start_p=1, start_q=1,
                        max_p=3, max_q=3,
                        seasonal=False,
                        stepwise=True,
                        suppress_warnings=True,
                        error_action='ignore'
                    )
                p, d, q = model_fit.order
            except Exception as e:
                messagebox.showerror("Error", f"An error occurred during ARIMA model fitting:\n{e}")
                return
            
            # Make predictions
            y_future = model_fit.predict(n_periods=num_predictions)
            
            # Prepare analysis text
            analysis_text = f"""
Statistical Analysis:

We performed an ARIMA({p},{d},{q}) analysis on the data to capture trends and patterns.

ARIMA Model Equation:
    y_t = c + φ₁ * y₍t₋₁₎ + θ₁ * ε₍t₋₁₎ + ε_t

Where:
    - y_t is the value at time t.
    - c is a constant term.
    - φ₁ is the autoregressive coefficient.
    - θ₁ is the moving average coefficient.
    - ε_t is the error term at time t.

Model Parameters:
"""
            # Get model parameters
            param_names = model_fit.arima_res_.param_names  # List of parameter names
            params = model_fit.arima_res_.params            # NumPy array of parameter values

            for param_name, param_value in zip(param_names, params):
                analysis_text += f"- {param_name}: {param_value:.4f}\n"
            
            # Include AIC and BIC
            aic = model_fit.aic()
            bic = model_fit.bic()
            analysis_text += f"\nModel Fit Statistics:\n- AIC: {aic:.4f}\n- BIC: {bic:.4f}\n"
            
            analysis_text += f"\nPredicted Next {num_predictions} Values:\n"
            for i, val in enumerate(y_future, start=1):
                analysis_text += f"Prediction {i}: {val:.4f}\n"
            
        elif selected_model == "Linear Regression":
            # Prepare data for Linear Regression
            x_reshape = x.reshape(-1, 1)
            # Fit Linear Regression model
            model = LinearRegression()
            model.fit(x_reshape, y)
            coeff = model.coef_[0]
            intercept = model.intercept_
            r_squared = model.score(x_reshape, y)
            
            # Make predictions
            x_future = np.arange(len(y), len(y) + num_predictions)
            x_future_reshape = x_future.reshape(-1, 1)
            y_future = model.predict(x_future_reshape)
            
            # Prepare analysis text
            analysis_text = f"""
Statistical Analysis:

We performed a Linear Regression on the data.

Linear Regression Equation:
y = {coeff:.4f} * x + {intercept:.4f}

Where:
- y is the predicted value.
- x is the time index.

Coefficient of Determination (R^2): {r_squared:.4f}

This model suggests that the data follows a linear trend.

Predicted Next {num_predictions} Values:
"""
            for i, val in enumerate(y_future, start=1):
                analysis_text += f"Prediction {i}: {val:.4f}\n"
        else:
            messagebox.showerror("Error", "Invalid model selection.")
            return
        
        # Display analysis text
        analysis_result_text.delete('1.0', tk.END)
        analysis_result_text.insert(tk.END, analysis_text)
        
        # Plot the data and predictions
        fig = plt.Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(x, y, 'bo-', label='Actual Data')
        
        if selected_model == "ARIMA":
            x_future = np.arange(len(y), len(y) + num_predictions)
            ax.plot(x_future, y_future, 'ro--', label='ARIMA Predictions')
        elif selected_model == "Linear Regression":
            x_future = np.arange(len(y) + num_predictions)
            y_line = model.predict(x_future.reshape(-1, 1))
            ax.plot(x_future, y_line, 'g-', label='Regression Line')
            ax.plot(x_future[-num_predictions:], y_future, 'ro--', label='Predictions')
        
        ax.set_title('Data Analysis and Predictions')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.legend()
        ax.grid(True)
        
        # Do not display the plot here; it will be displayed when 'Display Graph' button is clicked
        
    except Exception as e:
        import traceback
        traceback_str = traceback.format_exc()
        messagebox.showerror("Error", f"An error occurred during analysis:\n{traceback_str}")

def display_graph():
    global fig
    if 'fig' not in globals():
        messagebox.showwarning("Warning", "Please perform an analysis first.")
        return
    # Create a new window to display the graph
    graph_window = tk.Toplevel(root)
    graph_window.title("Graph")
    
    # Use a more robust method to locate the icon file
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diagram.png')
    if os.path.exists(icon_path):
        graph_window.iconphoto(False, tk.PhotoImage(file=icon_path))
    else:
        print(f"Warning: Icon file not found at {icon_path}")
    
    canvas_plot = FigureCanvasTkAgg(fig, master=graph_window)
    canvas_plot.draw()
    canvas_plot.get_tk_widget().pack()
def generate_report():
    global fig, analysis_text, data_series_used
    if 'analysis_text' not in globals() or 'fig' not in globals():
        messagebox.showwarning("Warning", "Please perform an analysis first.")
        return
    # Ask user where to save the report
    report_filename = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
    if not report_filename:
        return  # User cancelled the save dialog
    try:
        # Create a PDF canvas
        c = pdf_canvas.Canvas(report_filename, pagesize=letter)
        width, height = letter
        y_position = height - 50  # Start from top
        
        # Title
        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, y_position, "Statistical Data Analysis Report")
        y_position -= 40
        
        # Raw Data
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Raw Data:")
        y_position -= 20
        c.setFont("Helvetica", 10)
        data_text = data_series_used.to_string(index=True)
        text_lines = data_text.split('\n')
        for line in text_lines:
            c.drawString(50, y_position, line)
            y_position -= 15
            if y_position < 100:
                c.showPage()
                y_position = height - 50
        
        # Add space between Raw Data and Analysis Text
        y_position -= 20
        
        # Analysis Text
        c.setFont("Helvetica-Bold", 14)
        c.drawString(50, y_position, "Analysis:")
        y_position -= 20
        c.setFont("Helvetica", 10)
        analysis_lines = analysis_text.strip().split('\n')
        for line in analysis_lines:
            # Replace subscript and superscript numbers with their normal counterparts
            line = line.replace('₁', '1').replace('₍', '(').replace('₎', ')')
            line = line.replace('_t', '_t').replace('₋', '-')
            
            # Draw the modified line
            c.drawString(50, y_position, line)
            y_position -= 15
            if y_position < 100:
                c.showPage()
                y_position = height - 50
        
        # Graph
        # Save the figure to a bytes buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        img_reader = ImageReader(buf)
        img_width, img_height = fig.get_size_inches()
        img_width *= fig.dpi
        img_height *= fig.dpi
        y_position -= img_height + 10
        if y_position < 100:
            c.showPage()
            y_position = height - img_height - 50
        c.drawImage(img_reader, 50, y_position, width=500, preserveAspectRatio=True)
        buf.close()
        
        # Save the PDF
        c.save()
        messagebox.showinfo("Success", f"Report saved as {report_filename}")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred while generating the report:\n{e}")

def quit_application():
    root.destroy()

# Main application
root = tk.Tk()
root.title("Excel Statistical Data Analyzer - Andrew Thomas")
root.geometry("800x840") # size of window
# Use a more robust method to locate the icon file for compiling in autopytoexe
icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diagram.png')
if os.path.exists(icon_path):
    root.iconphoto(False, tk.PhotoImage(file=icon_path))
else:
    print(f"Warning: Icon file not found at {icon_path}")
root.resizable(False, False)  # Disable window resizing

excel_file_path = ""
df = pd.DataFrame()

# Variables for inputs
row_headers_var = tk.StringVar()
col_headers_var = tk.StringVar()
data_end_row_var = tk.StringVar()
data_end_col_var = tk.StringVar()
data_orientation_var = tk.StringVar(value="Rows")  # Default to Rows
data_position_var = tk.StringVar()
num_predictions_var = tk.StringVar(value="5")  # Default to 5 predictions
model_var = tk.StringVar(value="ARIMA")  # Default model is ARIMA

select_file_button = tk.Button(root, text="Select Excel File", command=select_file, 
                               bg='#90ee90', fg='black', 
                               activebackground='green', activeforeground='black',
                               relief=tk.RAISED, bd=3,
                               font=('Arial', 9, 'bold'))
select_file_button.pack(pady=5)

# Frame for data display with scrollbars
data_display_frame = tk.Frame(root)
data_display_frame.pack()

# Create canvas
canvas = tk.Canvas(data_display_frame, width=775, height=200)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

v_scrollbar = tk.Scrollbar(data_display_frame, orient=tk.VERTICAL, command=canvas.yview)
v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

h_scrollbar = tk.Scrollbar(data_display_frame, orient=tk.HORIZONTAL, command=canvas.xview)
h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

# Create a frame inside the canvas
data_frame = tk.Frame(canvas)
canvas.create_window((0,0), window=data_frame, anchor="nw")

# Update scrollregion when the canvas size changes
def on_configure(event):
    canvas.config(scrollregion=canvas.bbox("all"))

canvas.bind('<Configure>', on_configure)

# Frame for inputs
inputs_frame = tk.Frame(root)
inputs_frame.pack(pady=5)

# Row headers
tk.Label(inputs_frame, text="Row Headers start at row.. (blank if none):").grid(row=0, column=0, sticky='e')
row_headers_entry = tk.Entry(inputs_frame, textvariable=row_headers_var)
row_headers_entry.grid(row=0, column=1)

# Column headers
tk.Label(inputs_frame, text="Column Headers start at column.. (blank if none):").grid(row=1, column=0, sticky='e')
col_headers_entry = tk.Entry(inputs_frame, textvariable=col_headers_var)
col_headers_entry.grid(row=1, column=1)

# Data ends at row
tk.Label(inputs_frame, text="Data Ends at row.. (blank for all):").grid(row=2, column=0, sticky='e')
data_end_row_entry = tk.Entry(inputs_frame, textvariable=data_end_row_var)
data_end_row_entry.grid(row=2, column=1)

# Data ends at column
tk.Label(inputs_frame, text="Data Ends at column.. (blank for all):").grid(row=3, column=0, sticky='e')
data_end_col_entry = tk.Entry(inputs_frame, textvariable=data_end_col_var)
data_end_col_entry.grid(row=3, column=1)

# Analysis Options Frame
analysis_frame = tk.Frame(root)
analysis_frame.pack(pady=5)

# Data orientation (Rows or Columns)
tk.Label(analysis_frame, text="Data is in:").grid(row=0, column=0, sticky='e')
tk.Radiobutton(analysis_frame, text="Rows", variable=data_orientation_var, value="Rows").grid(row=0, column=1)
tk.Radiobutton(analysis_frame, text="Columns", variable=data_orientation_var, value="Columns").grid(row=0, column=2)

# Data position (Row or Column number)
tk.Label(analysis_frame, text="Row/Column number containing data to analyze:").grid(row=1, column=0, sticky='e')
data_position_entry = tk.Entry(analysis_frame, textvariable=data_position_var)
data_position_entry.grid(row=1, column=1)

# Number of predictions
tk.Label(analysis_frame, text="Number of predictions to make:").grid(row=2, column=0, sticky='e')
num_predictions_entry = tk.Entry(analysis_frame, textvariable=num_predictions_var)
num_predictions_entry.grid(row=2, column=1)

# Model selection
tk.Label(analysis_frame, text="Select Model:").grid(row=3, column=0, sticky='e')
tk.Radiobutton(analysis_frame, text="ARIMA", variable=model_var, value="ARIMA").grid(row=3, column=1)
tk.Radiobutton(analysis_frame, text="Linear Regression", variable=model_var, value="Linear Regression").grid(row=3, column=2)

# Frame for buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=5)

refresh_button = tk.Button(button_frame, text="Refresh Excel data", command=refresh_window,
                               bg='yellow', fg='black', 
                               activebackground='darkgoldenrod', activeforeground='black',
                               relief=tk.RAISED, bd=3,
                               font=('Arial', 9, 'bold'))
refresh_button.grid(row=0, column=0, padx=5)

analyze_button = tk.Button(button_frame, text="Analyze Data", command=analyze_data,
                               bg='#90ee90', fg='black', 
                               activebackground='green', activeforeground='black',
                               relief=tk.RAISED, bd=3,
                               font=('Arial', 9, 'bold'))
analyze_button.grid(row=0, column=1, padx=5)

# Display Graph Button
display_graph_button = tk.Button(button_frame, text="Display Graph", command=display_graph,
                               bg='yellow', fg='black', 
                               activebackground='darkgoldenrod', activeforeground='black',
                               relief=tk.RAISED, bd=3,
                               font=('Arial', 9, 'bold'))
display_graph_button.grid(row=0, column=2, padx=5)

# Generate Report Button
generate_report_button = tk.Button(button_frame, text="Save Report as PDF", command=generate_report,
                               bg='yellow', fg='black', 
                               activebackground='darkgoldenrod', activeforeground='black',
                               relief=tk.RAISED, bd=3,
                               font=('Arial', 9, 'bold'))
generate_report_button.grid(row=0, column=3, padx=5)

# Quit Button
quit_button = tk.Button(button_frame, text="Quit", command=quit_application, font=('Arial', 9, 'bold'))
quit_button.grid(row=0, column=4, padx=5)

# Create analysis_result_text widget
analysis_result_text = tk.Text(root, height=20, width=90)
analysis_result_text.pack(pady=10)

root.mainloop()
