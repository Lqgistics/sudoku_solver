import tkinter as tk

# Sample array of digits
digits = [
    [2, 5, 0, 0, 3, 0, 9, 0, 1],
    [0, 1, 0, 0, 0, 4, 0, 0, 0],
    [4, 0, 7, 0, 0, 0, 2, 0, 8],
    [0, 0, 5, 2, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 9, 8, 1, 0, 0],
    [0, 4, 0, 0, 0, 3, 0, 0, 0],
    [0, 0, 0, 3, 6, 0, 0, 7, 2],
    [0, 7, 0, 0, 0, 0, 0, 0, 3],
    [9, 0, 3, 0, 0, 0, 6, 0, 4]
]

# Create the main Tkinter window
root = tk.Tk()
root.title("Sudoku Grid")

# Function to display the digits
def display_array():
    for i in range(9):
        for j in range(9):
            digit = digits[i][j]
            label = tk.Label(frame, text=str(digit) if digit != 0 else "", width=2, relief="ridge")
            label.grid(row=i, column=j, padx=1, pady=1)

# Frame to contain the labels
frame = tk.Frame(root)
frame.pack()

# Button to trigger displaying the digits
display_button = tk.Button(root, text="Display Sudoku", command=display_array)
display_button.pack(pady=10)

root.mainloop()
