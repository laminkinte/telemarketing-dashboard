import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import warnings
import io
warnings.filterwarnings('ignore')

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Telemarketing Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS STYLING
# ============================================
st.markdown("""
<style>
    /* Main header */
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 3px solid #1E3A8A;
    }
    
    /* Sub headers */
    .sub-header {
        font-size: 1.8rem;
        color: #374151;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        margin: 1rem 0;
    }
    
    /* Warning boxes */
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #F59E0B;
        margin: 1rem 0;
    }
    
    /* Success boxes */
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #10B981;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #F8FAFC;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# DATA ANALYSIS FUNCTION
# ============================================
def analyze_telemarketing_data(df, start_date, end_date, selected_products):
    """
    Analyze transaction data with filters
    """
    try:
        # Display progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Clean column names
        status_text.text("Step 1/6: Cleaning column names...")
        df.columns = df.columns.str.strip()
        progress_bar.progress(15)
        
        # Step 2: Validate required columns
        status_text.text("Step 2/6: Validating data structure...")
        required_columns = ['User Identifier', 'Product Name', 'Service Name', 
                           'Created At', 'Entity Name', 'Full Name']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"❌ Missing required columns: {missing_columns}")
            st.error(f"Available columns: {list(df.columns)}")
            return None
        
        # Step 3: Clean and prepare data
        status_text.text("Step 3/6: Cleaning data...")
        df['Product Name'] = df['Product Name'].astype(str).str.strip()
        df['Entity Name'] = df['Entity Name'].astype(str).str.strip()
        df['User Identifier'] = pd.to_numeric(df['User Identifier'], errors='coerce')
        df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce', dayfirst=True)
        
        progress_bar.progress(30)
        
        # Step 4: Apply filters
        status_text.text("Step 4/6: Applying filters...")
        mask = (df['Created At'] >= pd.Timestamp(start_date)) & (df['Created At'] <= pd.Timestamp(end_date))
        filtered_df = df[mask].copy()
        
        if selected_products != 'All':
            filtered_df = filtered_df[filtered_df['Product Name'].isin(selected_products)]
        
        if filtered_df.empty:
            st.warning("⚠️ No data found for the selected filters!")
            return None
        
        # Step 5: Define categories
        status_text.text("Step 5/6: Categorizing products...")
        p2p_products = ['Internal Wallet Transfer (P2P)', 'Internal Wallet Transfer']
        international_remittance = 'International Remittance'
        other_services = [
            'Deposit', 'Scan To Withdraw Agent', 'Scan To Withdraw Customer', 
            'Scan To Send', 'Ticket'
        ]
        bill_payment_services = ['Bill Payment', 'Airtime Topup']
        
        progress_bar.progress(60)
        
        # Step 6: Filter customers only
        customer_df = filtered_df[
            (filtered_df['Entity Name'].str.contains('Customer', case=False, na=False)) |
            (filtered_df['Entity Name'].isna()) |
            (filtered_df['Entity Name'] == '')
        ].copy()
        
        # Get unique customers
        unique_customers = customer_df['User Identifier'].dropna().unique()
        
        # ============================================
        # REPORT 1: International remittance customers not using P2P
        # ============================================
        status_text.text("Step 6/6: Generating Report 1 - International customers...")
        intl_customers = filtered_df[
            filtered_df['Product Name'].str.contains('International Remittance', case=False, na=False)
        ]['User Identifier'].dropna().unique()
        
        p2p_customers = filtered_df[
            filtered_df['Product Name'].isin(p2p_products)
        ]['User Identifier'].dropna().unique()
        
        intl_not_p2p = [cust for cust in intl_customers if cust not in p2p_customers]
        
        # Create DataFrame for international remittance customers not using P2P
        customer_details = []
        for cust_id in intl_not_p2p[:1000]:  # Limit to first 1000 for performance
            cust_records = customer_df[customer_df['User Identifier'] == cust_id]
            
            if not cust_records.empty:
                valid_records = cust_records[cust_records['Full Name'].notna()]
                if not valid_records.empty:
                    cust_data = valid_records.iloc[0]
                    full_name = cust_data['Full Name']
                else:
                    full_name = 'Name not available'
                
                last_date = cust_records['Created At'].max()
                last_date_str = last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'Unknown'
                
                customer_details.append({
                    'User_ID': cust_id,
                    'Full_Name': full_name,
                    'Last_Transaction_Date': last_date_str,
                    'Total_Transactions': len(cust_records),
                    'International_Remittance_Count': len(cust_records[
                        cust_records['Product Name'].str.contains('International Remittance', case=False, na=False)
                    ])
                })
        
        if customer_details:
            report1_df = pd.DataFrame(customer_details)
        else:
            report1_df = pd.DataFrame(columns=['User_ID', 'Full_Name', 'Last_Transaction_Date', 
                                              'Total_Transactions', 'International_Remittance_Count'])
        
        # ============================================
        # REPORT 2: Non-international customers using other services but not P2P
        # ============================================
        other_services_customers = filtered_df[
            (filtered_df['Product Name'].isin(other_services)) |
            (filtered_df['Service Name'].isin(bill_payment_services))
        ]['User Identifier'].dropna().unique()
        
        domestic_other_not_p2p = [
            cust for cust in other_services_customers 
            if (cust not in intl_customers) and (cust not in p2p_customers)
        ]
        
        domestic_details = []
        for cust_id in domestic_other_not_p2p[:1000]:  # Limit to first 1000 for performance
            cust_records = customer_df[customer_df['User Identifier'] == cust_id]
            
            if not cust_records.empty:
                valid_records = cust_records[cust_records['Full Name'].notna()]
                if not valid_records.empty:
                    cust_data = valid_records.iloc[0]
                    full_name = cust_data['Full Name']
                else:
                    full_name = 'Name not available'
                
                last_date = cust_records['Created At'].max()
                last_date_str = last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'Unknown'
                
                services_used = cust_records['Product Name'].dropna().unique()
                other_services_used = []
                for service in services_used:
                    service_str = str(service)
                    if service_str in other_services:
                        other_services_used.append(service_str)
                    else:
                        service_records = cust_records[cust_records['Product Name'] == service]
                        if not service_records.empty:
                            service_name = service_records.iloc[0]['Service Name']
                            if pd.notna(service_name) and str(service_name) in bill_payment_services:
                                other_services_used.append(service_str)
                
                domestic_details.append({
                    'User_ID': cust_id,
                    'Full_Name': full_name,
                    'Last_Transaction_Date': last_date_str,
                    'Total_Transactions': len(cust_records),
                    'Other_Services_Used': ', '.join(other_services_used[:3]),  # Limit to first 3
                    'Other_Services_Count': len(other_services_used)
                })
        
        if domestic_details:
            report2_df = pd.DataFrame(domestic_details)
        else:
            report2_df = pd.DataFrame(columns=['User_ID', 'Full_Name', 'Last_Transaction_Date',
                                              'Total_Transactions', 'Other_Services_Used', 'Other_Services_Count'])
        
        # ============================================
        # REPORT 3: Daily Impact Report Template
        # ============================================
        dates = pd.date_range(start=start_date, end=end_date)
        
        impact_template = []
        for date in dates:
            date_str = date.strftime('%Y-%m-%d')
            impact_template.append({
                'Date': date_str,
                'Total_Customers_Called': 0,
                'Customers_Reached': 0,
                'Customers_Not_Reached': 0,
                'P2P_Adoption_After_Call': 0,
                'Conversion_Rate_%': 0.0,
                'Average_Transaction_Value': 0.0,
                'Notes': ''
            })
        
        impact_df = pd.DataFrame(impact_template)
        
        # ============================================
        # CALCULATE SUMMARY METRICS
        # ============================================
        conversion_potential = (len(intl_not_p2p) / len(intl_customers) * 100) if len(intl_customers) > 0 else 0
        
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        
        return {
            'intl_not_p2p': report1_df,
            'domestic_other_not_p2p': report2_df,
            'impact_template': impact_df,
            'summary': {
                'start_date': start_date,
                'end_date': end_date,
                'total_transactions': len(filtered_df),
                'unique_customers': len(unique_customers),
                'intl_customers': len(intl_customers),
                'intl_not_p2p': len(intl_not_p2p),
                'domestic_not_p2p': len(domestic_other_not_p2p),
                'p2p_users': len(p2p_customers),
                'conversion_potential': conversion_potential,
                'date_range_days': (end_date - start_date).days + 1,
                'products_analyzed': len(selected_products) if selected_products != 'All' else 'All'
            },
            'raw_data_stats': {
                'original_rows': len(df),
                'filtered_rows': len(filtered_df),
                'customer_rows': len(customer_df)
            }
        }
        
    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        return None

# ============================================
# MAIN STREAMLIT APPLICATION
# ============================================
def main():
    """Main Streamlit application"""
    
    # Title and description
    st.markdown('<h1 class="main-header">📊 Telemarketing Data Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
    <strong>📌 Purpose:</strong> Identify customers not using P2P services for targeted telemarketing campaigns<br>
    <strong>🎯 Goal:</strong> Increase P2P adoption among existing customers
    </div>
    """, unsafe_allow_html=True)
    
    # ============================================
    # SIDEBAR - File upload and filters
    # ============================================
    with st.sidebar:
        st.markdown("### 📁 Data Upload")
        st.markdown("---")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload your transaction data (CSV format)",
            type=['csv'],
            help="Upload a CSV file with transaction data. Required columns: User Identifier, Product Name, Service Name, Created At, Entity Name, Full Name"
        )
        
        if uploaded_file is not None:
            st.success("✅ File uploaded successfully!")
            
            # Preview uploaded file
            with st.expander("📋 Preview uploaded data", expanded=False):
                try:
                    df_preview = pd.read_csv(uploaded_file, nrows=5)
                    st.dataframe(df_preview)
                    st.caption(f"Columns: {', '.join(df_preview.columns.tolist())}")
                except:
                    st.warning("Cannot preview file")
            
            st.markdown("---")
            st.markdown("### ⚙️ Filter Settings")
            
            try:
                # Read the file once for filter options
                df = pd.read_csv(uploaded_file)
                
                # Date range selector
                st.markdown("#### 📅 Date Range")
                df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce', dayfirst=True)
                
                if df['Created At'].notna().any():
                    min_date = df['Created At'].min().date()
                    max_date = df['Created At'].max().date()
                else:
                    min_date = datetime.now().date() - timedelta(days=30)
                    max_date = datetime.now().date()
                    st.warning("⚠️ Date column has invalid values. Using default date range.")
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=min_date,
                        min_value=min_date,
                        max_value=max_date,
                        help="Select the start date for analysis"
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=max_date,
                        min_value=min_date,
                        max_value=max_date,
                        help="Select the end date for analysis"
                    )
                
                if start_date > end_date:
                    st.error("❌ Start date must be before end date!")
                    st.stop()
                
                # Product filter
                st.markdown("#### 📦 Product Filter")
                
                # Get unique products
                df['Product Name'] = df['Product Name'].astype(str).str.strip()
                all_products = sorted(df['Product Name'].dropna().unique().tolist())
                
                product_filter_option = st.radio(
                    "Product Selection Mode:",
                    ["All Products", "Select Specific Products"],
                    horizontal=True
                )
                
                if product_filter_option == "All Products":
                    selected_products = 'All'
                    st.info("📊 Analyzing all products")
                else:
                    selected_products = st.multiselect(
                        "Select products to include:",
                        all_products,
                        default=all_products[:5] if len(all_products) > 5 else all_products,
                        help="Select specific products to analyze. Use Ctrl+Click to select multiple."
                    )
                
                # Analysis button
                st.markdown("---")
                analyze_clicked = st.button(
                    "🚀 START ANALYSIS",
                    type="primary",
                    use_container_width=True,
                    help="Click to analyze data with selected filters"
                )
                
            except Exception as e:
                st.error(f"❌ Error reading file: {str(e)}")
                st.info("Please ensure your CSV file has the correct format and try again.")
                st.stop()
        else:
            # Show instructions when no file is uploaded
            st.markdown("""
            <div class="info-box">
            <h4>📝 How to use this dashboard:</h4>
            <ol>
            <li><strong>Upload</strong> your transaction CSV file</li>
            <li><strong>Set</strong> date range for analysis</li>
            <li><strong>Select</strong> products to include (or all)</li>
            <li><strong>Click</strong> "START ANALYSIS"</li>
            <li><strong>View</strong> results and download reports</li>
            </ol>
            
            <h4>📋 Required CSV columns:</h4>
            <ul>
            <li>User Identifier</li>
            <li>Product Name</li>
            <li>Service Name</li>
            <li>Created At (date)</li>
            <li>Entity Name</li>
            <li>Full Name</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
    
    # ============================================
    # MAIN CONTENT AREA
    # ============================================
    if uploaded_file is not None and 'analyze_clicked' in locals() and analyze_clicked:
        # Read the data
        df = pd.read_csv(uploaded_file)
        
        # Run analysis
        with st.spinner("🔍 Analyzing data... This may take a few moments."):
            results = analyze_telemarketing_data(df, start_date, end_date, selected_products)
        
        if results:
            # ============================================
            # SECTION 1: SUMMARY DASHBOARD
            # ============================================
            st.markdown('<h2 class="sub-header">📈 Performance Dashboard</h2>', unsafe_allow_html=True)
            
            # Create metric columns
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                <h3>📊 Transactions</h3>
                <h1>{results['summary']['total_transactions']:,}</h1>
                <p>Filtered period</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                <h3>👥 Unique Customers</h3>
                <h1>{results['summary']['unique_customers']:,}</h1>
                <p>Active in period</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                <h3>🌎 Int'l Not Using P2P</h3>
                <h1>{results['summary']['intl_not_p2p']:,}</h1>
                <p>Target for calling</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                conversion_color = "🟢" if results['summary']['conversion_potential'] > 30 else "🟡" if results['summary']['conversion_potential'] > 15 else "🔴"
                st.markdown(f"""
                <div class="metric-card">
                <h3>{conversion_color} Conversion Potential</h3>
                <h1>{results['summary']['conversion_potential']:.1f}%</h1>
                <p>Of international customers</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ============================================
            # SECTION 2: KEY INSIGHTS
            # ============================================
            st.markdown('<h2 class="sub-header">💡 Strategic Insights</h2>', unsafe_allow_html=True)
            
            insight_col1, insight_col2 = st.columns(2)
            
            with insight_col1:
                st.markdown("""
                <div class="info-box">
                <h4>🎯 Top Target Groups</h4>
                <ol>
                <li><strong>International Remittance Users</strong> ({intl_not_p2p} customers)
                <br><small>Highest potential for P2P adoption</small></li>
                <li><strong>Domestic Service Users</strong> ({domestic_not_p2p} customers)
                <br><small>Already using other services</small></li>
                </ol>
                </div>
                """.format(
                    intl_not_p2p=results['summary']['intl_not_p2p'],
                    domestic_not_p2p=results['summary']['domestic_not_p2p']
                ), unsafe_allow_html=True)
            
            with insight_col2:
                # Determine opportunity level
                if results['summary']['conversion_potential'] > 30:
                    opportunity_level = "HIGH"
                    opportunity_class = "success-box"
                    opportunity_emoji = "🚀"
                    recommendation = "Immediate calling campaign recommended"
                elif results['summary']['conversion_potential'] > 15:
                    opportunity_level = "MEDIUM"
                    opportunity_class = "warning-box"
                    opportunity_emoji = "📈"
                    recommendation = "Targeted calling campaign suggested"
                else:
                    opportunity_level = "LOW"
                    opportunity_class = "info-box"
                    opportunity_emoji = "📊"
                    recommendation = "Consider focusing on other segments"
                
                st.markdown(f"""
                <div class="{opportunity_class}">
                <h4>{opportunity_emoji} Opportunity Level: {opportunity_level}</h4>
                <p><strong>Conversion Potential:</strong> {results['summary']['conversion_potential']:.1f}%</p>
                <p><strong>Recommendation:</strong> {recommendation}</p>
                <p><strong>Target Pool Size:</strong> {results['summary']['intl_not_p2p'] + results['summary']['domestic_not_p2p']:,} customers</p>
                </div>
                """, unsafe_allow_html=True)
            
            # ============================================
            # SECTION 3: DETAILED REPORTS
            # ============================================
            st.markdown('<h2 class="sub-header">📋 Detailed Reports</h2>', unsafe_allow_html=True)
            
            # Create tabs for different reports
            tab1, tab2, tab3, tab4 = st.tabs([
                f"🌎 International ({len(results['intl_not_p2p'])})",
                f"🏠 Domestic ({len(results['domestic_other_not_p2p'])})",
                "📅 Impact Tracker",
                "📊 Data Summary"
            ])
            
            with tab1:
                st.markdown(f"""
                <div class="info-box">
                <h4>International Customers Not Using P2P</h4>
                <p>These customers use International Remittance but haven't used P2P services.
                They represent the highest potential for cross-selling.</p>
                <p><strong>Total:</strong> {len(results['intl_not_p2p'])} customers</p>
                </div>
                """, unsafe_allow_html=True)
                
                if not results['intl_not_p2p'].empty:
                    # Show data with pagination
                    page_size = 20
                    total_pages = max(1, len(results['intl_not_p2p']) // page_size)
                    
                    page = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=total_pages,
                        value=1,
                        key="intl_page"
                    )
                    
                    start_idx = (page - 1) * page_size
                    end_idx = min(page * page_size, len(results['intl_not_p2p']))
                    
                    st.dataframe(
                        results['intl_not_p2p'].iloc[start_idx:end_idx],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.caption(f"Showing records {start_idx + 1} to {end_idx} of {len(results['intl_not_p2p'])}")
                    
                    # Export options
                    col1, col2 = st.columns(2)
                    with col1:
                        csv1 = results['intl_not_p2p'].to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv1,
                            file_name=f"international_targets_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    with col2:
                        # Create a preview file with first 10 rows
                        preview_csv = results['intl_not_p2p'].head(10).to_csv(index=False)
                        st.download_button(
                            label="📋 Sample (10 rows)",
                            data=preview_csv,
                            file_name=f"international_sample_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.info("No international customers found without P2P usage in the selected period.")
            
            with tab2:
                st.markdown(f"""
                <div class="info-box">
                <h4>Domestic Customers Using Other Services (Not P2P)</h4>
                <p>These customers use various domestic services but haven't adopted P2P.
                They're already active users, making them good candidates for P2P introduction.</p>
                <p><strong>Total:</strong> {len(results['domestic_other_not_p2p'])} customers</p>
                </div>
                """, unsafe_allow_html=True)
                
                if not results['domestic_other_not_p2p'].empty:
                    # Show data with pagination
                    page_size = 20
                    total_pages = max(1, len(results['domestic_other_not_p2p']) // page_size)
                    
                    page = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=total_pages,
                        value=1,
                        key="domestic_page"
                    )
                    
                    start_idx = (page - 1) * page_size
                    end_idx = min(page * page_size, len(results['domestic_other_not_p2p']))
                    
                    st.dataframe(
                        results['domestic_other_not_p2p'].iloc[start_idx:end_idx],
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    st.caption(f"Showing records {start_idx + 1} to {end_idx} of {len(results['domestic_other_not_p2p'])}")
                    
                    # Export options
                    col1, col2 = st.columns(2)
                    with col1:
                        csv2 = results['domestic_other_not_p2p'].to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv2,
                            file_name=f"domestic_targets_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    with col2:
                        preview_csv = results['domestic_other_not_p2p'].head(10).to_csv(index=False)
                        st.download_button(
                            label="📋 Sample (10 rows)",
                            data=preview_csv,
                            file_name=f"domestic_sample_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                else:
                    st.info("No domestic customers found using other services without P2P in the selected period.")
            
            with tab3:
                st.markdown("""
                <div class="info-box">
                <h4>Daily Impact Tracking Template</h4>
                <p>Use this template to track your daily calling campaign results.
                Update the metrics after each calling session to measure effectiveness.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.dataframe(results['impact_template'], use_container_width=True)
                
                # Editable impact tracker
                st.markdown("### 📝 Update Impact Tracker")
                st.info("Edit the values below and click 'Update Tracker' to modify the template")
                
                edit_col1, edit_col2, edit_col3 = st.columns(3)
                with edit_col1:
                    edit_date = st.selectbox(
                        "Select Date to Edit",
                        options=results['impact_template']['Date'].tolist()
                    )
                
                with edit_col2:
                    edit_metric = st.selectbox(
                        "Select Metric to Edit",
                        options=['Total_Customers_Called', 'Customers_Reached', 'Customers_Not_Reached',
                                'P2P_Adoption_After_Call', 'Conversion_Rate_%', 'Average_Transaction_Value', 'Notes']
                    )
                
                with edit_col3:
                    if edit_metric == 'Notes':
                        new_value = st.text_input("New Value", value="")
                    elif edit_metric == 'Conversion_Rate_%':
                        new_value = st.number_input("New Value", min_value=0.0, max_value=100.0, value=0.0, step=0.1)
                    else:
                        new_value = st.number_input("New Value", min_value=0, value=0, step=1)
                
                if st.button("Update Tracker", type="secondary"):
                    idx = results['impact_template'][results['impact_template']['Date'] == edit_date].index[0]
                    results['impact_template'].at[idx, edit_metric] = new_value
                    st.success(f"Updated {edit_date} - {edit_metric} to {new_value}")
                    st.rerun()
                
                # Export impact template
                csv3 = results['impact_template'].to_csv(index=False)
                st.download_button(
                    label="📥 Download Impact Template",
                    data=csv3,
                    file_name=f"impact_tracker_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with tab4:
                st.markdown("### 📊 Data Summary")
                
                summary_col1, summary_col2 = st.columns(2)
                
                with summary_col1:
                    st.markdown("#### 📈 Analysis Statistics")
                    summary_data = pd.DataFrame({
                        'Metric': [
                            'Analysis Period',
                            'Days Analyzed',
                            'Products Included',
                            'Original Rows',
                            'Filtered Rows',
                            'Customer Rows',
                            'P2P Users',
                            'Calling Targets'
                        ],
                        'Value': [
                            f"{results['summary']['start_date'].strftime('%Y-%m-%d')} to {results['summary']['end_date'].strftime('%Y-%m-%d')}",
                            results['summary']['date_range_days'],
                            results['summary']['products_analyzed'],
                            f"{results['raw_data_stats']['original_rows']:,}",
                            f"{results['raw_data_stats']['filtered_rows']:,}",
                            f"{results['raw_data_stats']['customer_rows']:,}",
                            f"{results['summary']['p2p_users']:,}",
                            f"{results['summary']['intl_not_p2p'] + results['summary']['domestic_not_p2p']:,}"
                        ]
                    })
                    st.dataframe(summary_data, use_container_width=True, hide_index=True)
                
                with summary_col2:
                    st.markdown("#### 📋 Report Contents")
                    report_contents = pd.DataFrame({
                        'Report': [
                            'International Target List',
                            'Domestic Target List',
                            'Impact Tracker Template'
                        ],
                        'Records': [
                            len(results['intl_not_p2p']),
                            len(results['domestic_other_not_p2p']),
                            len(results['impact_template'])
                        ],
                        'Description': [
                            'Customers using international remittance but not P2P',
                            'Domestic customers using other services but not P2P',
                            'Daily tracking template for calling campaigns'
                        ]
                    })
                    st.dataframe(report_contents, use_container_width=True, hide_index=True)
            
            # ============================================
            # SECTION 4: EXPORT AND NEXT STEPS
            # ============================================
            st.markdown('<h2 class="sub-header">🚀 Export & Action Plan</h2>', unsafe_allow_html=True)
            
            # Export full report
            st.markdown("### 💾 Export Complete Report")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Create Excel report
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                    results['intl_not_p2p'].to_excel(writer, sheet_name='International_Targets', index=False)
                    results['domestic_other_not_p2p'].to_excel(writer, sheet_name='Domestic_Targets', index=False)
                    results['impact_template'].to_excel(writer, sheet_name='Impact_Tracker', index=False)
                    
                    # Summary sheet
                    summary_df = pd.DataFrame({
                        'Analysis_Period': [f"{results['summary']['start_date']} to {results['summary']['end_date']}"],
                        'Total_Transactions': [results['summary']['total_transactions']],
                        'Unique_Customers': [results['summary']['unique_customers']],
                        'International_Targets': [results['summary']['intl_not_p2p']],
                        'Domestic_Targets': [results['summary']['domestic_not_p2p']],
                        'Conversion_Potential_%': [results['summary']['conversion_potential']],
                        'Generated_On': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                    })
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                excel_data = output.getvalue()
                
                st.download_button(
                    label="📥 Download Full Report (Excel)",
                    data=excel_data,
                    file_name=f"telemarketing_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
            with col2:
                # Quick summary PDF (text-based)
                summary_text = f"""
                TELEMARKETING ANALYSIS REPORT
                =============================
                
                Report Period: {results['summary']['start_date']} to {results['summary']['end_date']}
                Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                
                KEY METRICS:
                - Total Transactions: {results['summary']['total_transactions']:,}
                - Unique Customers: {results['summary']['unique_customers']:,}
                - International Targets: {results['summary']['intl_not_p2p']:,}
                - Domestic Targets: {results['summary']['domestic_not_p2p']:,}
                - Conversion Potential: {results['summary']['conversion_potential']:.1f}%
                
                RECOMMENDED ACTIONS:
                1. Start calling international customers first
                2. Allocate {results['summary']['intl_not_p2p'] // 5} calls per day for 5 days
                3. Track results using the impact tracker
                4. Measure P2P adoption weekly
                
                """
                
                st.download_button(
                    label="📄 Download Summary (TXT)",
                    data=summary_text,
                    file_name=f"summary_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            
            with col3:
                st.markdown("""
                <div class="info-box">
                <h4>📞 Quick Actions</h4>
                <ul>
                <li>Copy target lists to your CRM</li>
                <li>Schedule calling sessions</li>
                <li>Assign teams to segments</li>
                <li>Set daily calling goals</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Action plan
            st.markdown("### 🎯 Recommended Action Plan")
            
            plan_col1, plan_col2 = st.columns(2)
            
            with plan_col1:
                st.markdown("""
                <div class="success-box">
                <h4>📅 Week 1: International Campaign</h4>
                <ul>
                <li><strong>Day 1-2:</strong> Call top 20% of international targets</li>
                <li><strong>Day 3-4:</strong> Follow up on interested customers</li>
                <li><strong>Day 5:</strong> Review conversion rates</li>
                <li><strong>Target:</strong> 15% P2P adoption rate</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            with plan_col2:
                st.markdown("""
                <div class="warning-box">
                <h4>📅 Week 2: Domestic Campaign</h4>
                <ul>
                <li><strong>Day 1-3:</strong> Contact domestic service users</li>
                <li><strong>Day 4-5:</strong> Offer P2P tutorials/demos</li>
                <li><strong>Target:</strong> 10% P2P adoption rate</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Final tips
            st.markdown("""
            <div class="info-box">
            <h4>💡 Pro Tips for Success</h4>
            <ol>
            <li><strong>Personalize calls:</strong> Reference customers' recent transactions</li>
            <li><strong>Track everything:</strong> Use the impact tracker daily</li>
            <li><strong>Measure ROI:</strong> Calculate value of converted customers</li>
            <li><strong>Iterate:</strong> Refine scripts based on what works</li>
            <li><strong>Automate:</strong> Consider SMS/email follow-ups</li>
            </ol>
            </div>
            """, unsafe_allow_html=True)

# ============================================
# RUN THE APPLICATION
# ============================================
if __name__ == "__main__":
    main()
