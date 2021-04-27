import streamlit as st

class latex_equation_numbering():
    # The styling options for display and inline equation numbers can be 'plain', 'bold', or 'italic'
    def __init__(self, display_inline_number_style=('plain', 'plain')):
        self.equation_dict = {}
        self.counter = 1
        self.display_number_style = display_inline_number_style[0]
        self.inline_number_style = display_inline_number_style[1]
    
    # Method for adding new display equations to the dictionary, with a choice of displaying an equation number for future reference
    def add_equation(self, label, equation, numbered=True):
        # Add an item to the equation_dict
        # Each equation needs a raw string for the latex, the equation number, and whether to display the number
        # If 'numbered' is False, the equation will still have a number assigned, but it will not be incremented
        self.equation_dict[label] = {'eq': equation, 'number': self.counter, 'numbered': numbered}

        if numbered:
            self.counter += 1 # Increment the counter

    # Method for displaying an equation 
    def display_equation(self, label):

        if self.equation_dict[label]['numbered']:
            self.latex_numbered_equation(label) # Display the equation
            
        else:
            self.latex_unnumbered_equation(label) # Display the equation and skip incrementing the counter

    # Method for displaying a numbered equation
    def latex_numbered_equation(self, label):

        # Get equation number:
        eqnum = self.equation_dict[label]['number']

        # Create three columns, with the center one being wider
        col1, col2, col3 = st.beta_columns([1,5,1])
        # Add an invisible HTML div to the first column with the id equal to the equation label, insert the equation in the center column, and the label in the third.
        # The displayed number style is given by the value of self.display_number_style
        # We use fr=strings to insert our arguments into raw strings.

        # Empty column with empty div to link to equation
        with col1:
            st.markdown(f"<div id='{label}'></div>", unsafe_allow_html=True)
        
        # write the equation
        with col2:
            st.write(
                fr'''
                $$
                {self.equation_dict[label]['eq']}
                $$
                '''
            )
        
        
        with col3:
            
            if self.display_number_style == 'plain':
                st.write(fr'''
                    $$
                    {eqnum}
                    $$
                    '''
                )

            elif self.display_number_style == 'bold':
                st.write(fr'''
                    $$
                    \textbf{eqnum}
                    $$
                    '''
                )
            

            elif self.display_number_style == 'italic':
                st.write(fr'''
                    $$
                    \textit{eqnum}
                    $$
                    '''
                )
            
    
    # Method for displaying an unnumbered equation
    def latex_unnumbered_equation(self, label):
        # Display the equation
        st.write(
                fr'''
                $$
                {self.equation_dict[label]['eq']}
                $$
                '''
            )
    
    # Method for generating an inline reference to displayed equations
    def eqref(self, label, linked=True):

        # Get equation number:
        eqnum = self.equation_dict[label]['number']

        if linked:
            if self.inline_number_style == 'plain':
                return fr"<a href='#{label}' style='text-decoration:none; color:inherit'>{eqnum}</a>"

            elif self.inline_number_style == 'bold':
                return fr"<a href='#{label}' style='text-decoration:none; color:inherit; font-weight:bold'>{eqnum}</a>"

            elif self.inline_number_style == 'italic':
                return fr"<a href='#{label}' style='text-decoration:none; color:inherit; font-style: italic;'>{eqnum}</a>"

        else:
            if self.inline_number_style == 'plain':
                return fr"{eqnum}"
            
            elif self.inline_number_style == 'bold':
                return fr"**{eqnum}**"
            
            elif self.inline_number_style == 'italic':
                return fr"_{eqnum}_"
