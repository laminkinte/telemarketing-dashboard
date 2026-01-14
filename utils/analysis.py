"""
Core analysis functions with Excel sheet-wise export
Fixed to handle comma-separated numbers
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import re
import warnings
warnings.filterwarnings('ignore')

def clean_numeric_string(value):
    """Clean numeric strings with commas and other formatting"""
    if pd.isna(value):
        return np.nan
    
    # Convert to string if not already
    str_value = str(value).strip()
    
    # Remove currency symbols and spaces
    str_value = re.sub(r'[^\d.,-]', '', str_value)
    
    # Remove commas (thousands separators)
    if ',' in str_value and '.' in str_value:
        # Handle cases like "1,000.00"
        str_value = str_value.replace(',', '')
    elif ',' in str_value and '.' not in str_value:
        # Handle cases like "1,000"
        str_value = str_value.replace(',', '')
    
    # Convert to float
    try:
        return float(str_value)
    except:
        return np.nan

def clean_and_prepare_data(df):
    """Clean and prepare transaction data with robust numeric handling"""
    df_clean = df.copy()
    
    # Clean column names
    df_clean.columns = [col.strip() for col in df_clean.columns]
    
    # Convert User Identifier to numeric (handle errors)
    if 'User Identifier' in df_clean.columns:
        df_clean['User Identifier'] = pd.to_numeric(df_clean['User Identifier'], errors='coerce')
    
    # Clean text columns
    text_cols = ['Product Name', 'Service Name', 'Entity Name', 'Full Name', 'Status']
    for col in text_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
    
    # Clean numeric columns
    numeric_cols = ['Amount', 'Before Balance', 'After Balance']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].apply(clean_numeric_string)
    
    # Parse dates with multiple format attempts
    if 'Created At' in df_clean.columns:
        # Try multiple date formats
        date_formats = [
            '%d/%m/%Y %H:%M',        # 11/01/2026 23:59
            '%d/%m/%Y %H:%M:%S',     # 11/01/2026 23:59:00
            '%Y-%m-%d %H:%M:%S',     # 2026-01-11 23:59:00
            '%m/%d/%Y %H:%M',        # 01/11/2026 23:59
            '%Y/%m/%d %H:%M:%S',     # 2026/01/11 23:59:00
            '%d-%m-%Y %H:%M',        # 11-01-2026 23:59
            '%Y-%m-%d',              # 2026-01-11
            '%d/%m/%Y',              # 11/01/2026
        ]
        
        for fmt in date_formats:
            try:
                parsed_dates = pd.to_datetime(df_clean['Created At'], format=fmt, errors='coerce')
                if parsed_dates.notna().any():
                    df_clean['Created At'] = parsed_dates
                    break
            except:
                continue
        
        # Final fallback
        if df_clean['Created At'].isna().all() or df_clean['Created At'].dtype == 'object':
            df_clean['Created At'] = pd.to_datetime(df_clean['Created At'], errors='coerce')
    
    return df_clean

def identify_product_categories(product_name):
    """Categorize products based on name"""
    if not isinstance(product_name, str):
        return 'Other'
    
    product_lower = product_name.lower()
    
    # P2P Products
    if any(keyword in product_lower for keyword in ['internal wallet transfer', 'p2p', 'wallet transfer']):
        return 'P2P'
    
    # International Remittance
    elif any(keyword in product_lower for keyword in ['international remittance', 'international transfer', 'remittance']):
        return 'International Remittance'
    
    # Other Services
    elif any(keyword in product_lower for keyword in ['deposit', 'scan', 'ticket']):
        return 'Other Service'
    
    # Bill Payment & Airtime
    elif any(keyword in product_lower for keyword in ['bill', 'airtime', 'nawec', 'comium', 'africell', 'qcell']):
        return 'Bill Payment/Airtime'
    
    else:
        return 'Other'

def analyze_telemarketing_data(df, analysis_period="Last 7 days"):
    """
    Main analysis function with robust error handling
    Returns dictionary with all reports
    """
    try:
        # Clean data
        df_clean = clean_and_prepare_data(df)
        
        # Check required columns
        required_cols = ['User Identifier', 'Product Name', 'Created At', 'Entity Name']
        missing_cols = [col for col in required_cols if col not in df_clean.columns]
        
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return {
                'intl_not_p2p': pd.DataFrame({'Error': [f'Missing columns: {", ".join(missing_cols)}']}),
                'domestic_other_not_p2p': pd.DataFrame(),
                'impact_template': pd.DataFrame(),
                'detailed_analysis': pd.DataFrame(),
                'summary': {}
            }
        
        # Add product category column
        df_clean['Product_Category'] = df_clean['Product Name'].apply(identify_product_categories)
        
        # Filter by date range
        if df_clean['Created At'].notna().any():
            end_date = df_clean['Created At'].max()
            
            if analysis_period == "Last 7 days":
                start_date = end_date - timedelta(days=7)
            elif analysis_period == "Last 30 days":
                start_date = end_date - timedelta(days=30)
            elif analysis_period == "Last 90 days":
                start_date = end_date - timedelta(days=90)
            else:  # All data
                start_date = df_clean['Created At'].min()
            
            date_mask = (df_clean['Created At'] >= start_date) & (df_clean['Created At'] <= end_date)
            df_period = df_clean[date_mask].copy()
        else:
            df_period = df_clean.copy()
            start_date = None
            end_date = None
        
        # Filter for customers only
        customer_mask = (
            df_period['Entity Name'].str.contains('Customer', case=False, na=False) |
            df_period['Entity Name'].isna() |
            (df_period['Entity Name'] == '') |
            (df_period['Entity Name'].str.contains('customer', case=False, na=False))
        )
        df_customers = df_period[customer_mask].copy()
        
        # Get unique customers
        unique_customers = df_customers['User Identifier'].dropna().unique()
        
        # REPORT 1: International customers not using P2P
        intl_customers = df_customers[
            df_customers['Product_Category'] == 'International Remittance'
        ]['User Identifier'].dropna().unique()
        
        p2p_customers = df_customers[
            df_customers['Product_Category'] == 'P2P'
        ]['User Identifier'].dropna().unique()
        
        intl_not_p2p = [cust for cust in intl_customers if cust not in p2p_customers]
        
        # Build detailed report for international customers
        intl_report_data = []
        for cust_id in intl_not_p2p[:5000]:  # Limit for performance
            cust_data = df_customers[df_customers['User Identifier'] == cust_id]
            if not cust_data.empty:
                # Get customer info
                if 'Full Name' in cust_data.columns:
                    name_data = cust_data[cust_data['Full Name'].notna() & (cust_data['Full Name'] != 'nan')]
                    full_name = name_data['Full Name'].iloc[0] if not name_data.empty else 'Unknown'
                else:
                    full_name = 'Unknown'
                
                # Get transaction info
                last_date = cust_data['Created At'].max()
                last_date_str = last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'Unknown'
                
                # Count transactions
                total_tx = len(cust_data)
                intl_tx = len(cust_data[cust_data['Product_Category'] == 'International Remittance'])
                
                # Get amounts if available
                if 'Amount' in cust_data.columns:
                    amount_data = cust_data['Amount'].dropna()
                    avg_amount = amount_data.mean() if len(amount_data) > 0 else 0
                    total_amount = amount_data.sum() if len(amount_data) > 0 else 0
                else:
                    avg_amount = 0
                    total_amount = 0
                
                # Calculate days since last transaction
                days_since = (datetime.now() - last_date).days if pd.notna(last_date) and last_date <= datetime.now() else 'Unknown'
                
                intl_report_data.append({
                    'Customer_ID': int(cust_id) if pd.notna(cust_id) else 'Unknown',
                    'Full_Name': full_name,
                    'Last_Transaction_Date': last_date_str,
                    'Total_Transactions': total_tx,
                    'International_Transactions': intl_tx,
                    'Avg_Transaction_Amount': round(avg_amount, 2),
                    'Total_Transaction_Amount': round(total_amount, 2),
                    'Days_Since_Last_Tx': days_since
                })
        
        intl_report_df = pd.DataFrame(intl_report_data) if intl_report_data else pd.DataFrame(
            columns=['Customer_ID', 'Full_Name', 'Last_Transaction_Date', 'Total_Transactions',
                    'International_Transactions', 'Avg_Transaction_Amount', 
                    'Total_Transaction_Amount', 'Days_Since_Last_Tx']
        )
        
        # REPORT 2: Domestic customers using other services but not P2P
        other_customers = df_customers[
            (df_customers['Product_Category'] == 'Other Service') |
            (df_customers['Product_Category'] == 'Bill Payment/Airtime')
        ]['User Identifier'].dropna().unique()
        
        domestic_not_p2p = [
            cust for cust in other_customers 
            if (cust not in intl_customers) and (cust not in p2p_customers)
        ]
        
        # Build detailed report for domestic customers
        domestic_report_data = []
        for cust_id in domestic_not_p2p[:5000]:  # Limit for performance
            cust_data = df_customers[df_customers['User Identifier'] == cust_id]
            if not cust_data.empty:
                # Get customer info
                if 'Full Name' in cust_data.columns:
                    name_data = cust_data[cust_data['Full Name'].notna() & (cust_data['Full Name'] != 'nan')]
                    full_name = name_data['Full Name'].iloc[0] if not name_data.empty else 'Unknown'
                else:
                    full_name = 'Unknown'
                
                # Get transaction info
                last_date = cust_data['Created At'].max()
                last_date_str = last_date.strftime('%Y-%m-%d') if pd.notna(last_date) else 'Unknown'
                
                # Get product usage
                product_counts = cust_data['Product_Category'].value_counts()
                services_used = []
                for product_cat in ['Other Service', 'Bill Payment/Airtime']:
                    if product_cat in product_counts:
                        services_used.append(f"{product_cat}: {product_counts[product_cat]}")
                
                # Get total amount
                if 'Amount' in cust_data.columns:
                    amount_data = cust_data['Amount'].dropna()
                    total_amount = amount_data.sum() if len(amount_data) > 0 else 0
                else:
                    total_amount = 0
                
                # Calculate days since last transaction
                days_since = (datetime.now() - last_date).days if pd.notna(last_date) and last_date <= datetime.now() else 'Unknown'
                
                domestic_report_data.append({
                    'Customer_ID': int(cust_id) if pd.notna(cust_id) else 'Unknown',
                    'Full_Name': full_name,
                    'Last_Transaction_Date': last_date_str,
                    'Total_Transactions': len(cust_data),
                    'Services_Used': '; '.join(services_used[:3]),
                    'Total_Amount': round(total_amount, 2),
                    'Days_Since_Last_Tx': days_since
                })
        
        domestic_report_df = pd.DataFrame(domestic_report_data) if domestic_report_data else pd.DataFrame(
            columns=['Customer_ID', 'Full_Name', 'Last_Transaction_Date', 'Total_Transactions',
                    'Services_Used', 'Total_Amount', 'Days_Since_Last_Tx']
        )
        
        # REPORT 3: Impact template
        if start_date and end_date:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            dates = pd.date_range(end=datetime.now(), periods=7, freq='D')
        
        impact_data = []
        for date in dates:
            impact_data.append({
                'Date': date.strftime('%Y-%m-%d'),
                'Customers_Called': 0,
                'Customers_Reached': 0,
                'Customers_Not_Reached': 0,
                'P2P_Conversions': 0,
                'Conversion_Rate_%': 0.0,
                'Avg_Call_Duration_min': 0,
                'Follow_Ups_Required': 0,
                'Notes': ''
            })
        
        impact_df = pd.DataFrame(impact_data)
        
        # REPORT 4: Detailed analysis
        detailed_analysis = create_detailed_analysis(
            df_customers, intl_customers, p2p_customers, 
            intl_not_p2p, domestic_not_p2p
        )
        
        # Summary statistics
        total_intl = len(intl_customers)
        total_p2p = len(p2p_customers)
        total_domestic = len(domestic_not_p2p)
        total_other = len(other_customers)
        
        conversion_potential = (len(intl_not_p2p) / max(total_intl, 1)) * 100 if total_intl > 0 else 0
        
        summary = {
            'Report_Period': f"{start_date.strftime('%Y-%m-%d') if start_date else 'N/A'} to {end_date.strftime('%Y-%m-%d') if end_date else 'N/A'}",
            'Total_Transactions_Analyzed': len(df_period),
            'Total_Unique_Customers': len(unique_customers),
            'International_Customers': total_intl,
            'International_Customers_Not_Using_P2P': len(intl_not_p2p),
            'Domestic_Customers_Not_Using_P2P': total_domestic,
            'P2P_Users': total_p2p,
            'Other_Service_Users': total_other,
            'Conversion_Potential_%': round(conversion_potential, 2),
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Data_Source': 'Transaction CSV Analysis'
        }
        
        return {
            'intl_not_p2p': intl_report_df,
            'domestic_other_not_p2p': domestic_report_df,
            'impact_template': impact_df,
            'detailed_analysis': detailed_analysis,
            'summary': summary,
            'raw_data_sample': df_clean.head(100)  # Add sample for reference
        }
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Detailed error: {error_details}")
        raise Exception(f"Analysis error: {str(e)}")

def create_detailed_analysis(df_customers, intl_customers, p2p_customers, 
                           intl_not_p2p, domestic_not_p2p):
    """Create detailed analysis DataFrame"""
    
    analysis_data = []
    
    # 1. Customer segments
    total_unique = len(df_customers['User Identifier'].dropna().unique())
    
    segments = {
        'International P2P Users': len(set(intl_customers) & set(p2p_customers)),
        'International Non-P2P Users': len(intl_not_p2p),
        'Domestic Non-P2P Users': len(domestic_not_p2p),
        'P2P Only Users': len(set(p2p_customers) - set(intl_customers)),
        'Other Active Users': total_unique - len(set(intl_customers) | set(p2p_customers) | set(domestic_not_p2p))
    }
    
    for segment, count in segments.items():
        percentage = (count / max(total_unique, 1)) * 100
        analysis_data.append({
            'Category': 'Customer Segments',
            'Metric': segment,
            'Value': count,
            'Percentage': round(percentage, 2)
        })
    
    # 2. Transaction patterns
    if 'Amount' in df_customers.columns:
        amounts = df_customers['Amount'].dropna()
        if len(amounts) > 0:
            analysis_data.append({
                'Category': 'Transaction Patterns',
                'Metric': 'Average Transaction Amount',
                'Value': round(amounts.mean(), 2),
                'Percentage': None
            })
            analysis_data.append({
                'Category': 'Transaction Patterns',
                'Metric': 'Maximum Transaction Amount',
                'Value': round(amounts.max(), 2),
                'Percentage': None
            })
            analysis_data.append({
                'Category': 'Transaction Patterns',
                'Metric': 'Total Transaction Volume',
                'Value': round(amounts.sum(), 2),
                'Percentage': None
            })
    
    # 3. Time-based metrics
    if 'Created At' in df_customers.columns and df_customers['Created At'].notna().any():
        df_customers['Transaction_Date'] = df_customers['Created At'].dt.date
        daily_tx = df_customers.groupby('Transaction_Date').size()
        
        if len(daily_tx) > 0:
            analysis_data.append({
                'Category': 'Time Patterns',
                'Metric': 'Average Daily Transactions',
                'Value': round(daily_tx.mean(), 2),
                'Percentage': None
            })
            analysis_data.append({
                'Category': 'Time Patterns',
                'Metric': 'Peak Day Transactions',
                'Value': daily_tx.max(),
                'Percentage': None
            })
    
    return pd.DataFrame(analysis_data)

def generate_excel_report(results):
    """
    Generate Excel report with multiple sheets
    Returns: BytesIO object with Excel file
    """
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Sheet 1: Executive Summary
        if 'summary' in results:
            summary_df = pd.DataFrame({
                'Key Metric': list(results['summary'].keys()),
                'Value': list(results['summary'].values())
            })
            summary_df.to_excel(writer, sheet_name='Executive_Summary', index=False)
            worksheet = writer.sheets['Executive_Summary']
            worksheet.column_dimensions['A'].width = 35
            worksheet.column_dimensions['B'].width = 40
        
        # Sheet 2: International Customers Not Using P2P
        if 'intl_not_p2p' in results and not results['intl_not_p2p'].empty:
            results['intl_not_p2p'].to_excel(
                writer, 
                sheet_name='International_Not_P2P', 
                index=False
            )
            worksheet = writer.sheets['International_Not_P2P']
            # Set column widths
            column_widths = {
                'A': 15, 'B': 25, 'C': 20, 'D': 15, 'E': 20,
                'F': 20, 'G': 20, 'H': 20
            }
            for col, width in column_widths.items():
                worksheet.column_dimensions[col].width = width
        
        # Sheet 3: Domestic Customers Not Using P2P
        if 'domestic_other_not_p2p' in results and not results['domestic_other_not_p2p'].empty:
            results['domestic_other_not_p2p'].to_excel(
                writer, 
                sheet_name='Domestic_Not_P2P', 
                index=False
            )
            worksheet = writer.sheets['Domestic_Not_P2P']
            column_widths = {'A': 15, 'B': 25, 'C': 20, 'D': 15, 'E': 30, 'F': 15, 'G': 20}
            for col, width in column_widths.items():
                worksheet.column_dimensions[col].width = width
        
        # Sheet 4: Detailed Analysis
        if 'detailed_analysis' in results and not results['detailed_analysis'].empty:
            results['detailed_analysis'].to_excel(
                writer, 
                sheet_name='Detailed_Analysis', 
                index=False
            )
            worksheet = writer.sheets['Detailed_Analysis']
            worksheet.column_dimensions['A'].width = 20
            worksheet.column_dimensions['B'].width = 30
            worksheet.column_dimensions['C'].width = 15
            worksheet.column_dimensions['D'].width = 15
        
        # Sheet 5: Impact Tracking Template
        if 'impact_template' in results:
            results['impact_template'].to_excel(
                writer, 
                sheet_name='Impact_Tracking', 
                index=False
            )
            worksheet = writer.sheets['Impact_Tracking']
            worksheet.column_dimensions['A'].width = 15
            worksheet.column_dimensions['B'].width = 15
            worksheet.column_dimensions['C'].width = 15
            worksheet.column_dimensions['D'].width = 20
            worksheet.column_dimensions['E'].width = 15
            worksheet.column_dimensions['F'].width = 15
            worksheet.column_dimensions['G'].width = 20
            worksheet.column_dimensions['H'].width = 15
            worksheet.column_dimensions['I'].width = 30
        
        # Sheet 6: Calling Recommendations
        recommendations = pd.DataFrame({
            'Priority_Level': ['High', 'High', 'Medium', 'Medium', 'Low'],
            'Customer_Group': [
                'Recent International Remittance Users (Last 7 days)',
                'High Value International Users (Avg. Amount > $500)',
                'Active Domestic Service Users (Multiple transactions)',
                'Historical International Users (Last 30-90 days)',
                'Inactive Customers (No transactions in 90+ days)'
            ],
            'Recommended_Action': [
                'Call within 24-48 hours. Highlight P2P speed and lower fees.',
                'Schedule appointment. Discuss bulk transfer benefits.',
                'Educate during next service interaction. Offer P2P demo.',
                'Email/SMS campaign first, then follow-up call.',
                'Consider win-back campaign with special incentives.'
            ],
            'Expected_Conversion_Rate': ['30-40%', '25-35%', '15-25%', '10-20%', '5-15%'],
            'Key_Talking_Points': [
                'Instant transfers vs. traditional banking delays',
                'Cost savings on frequent transfers',
                'Convenience for local transactions',
                'Re-engagement offer: First P2P transfer fee waived',
                'Limited time promotion for returning customers'
            ]
        })
        recommendations.to_excel(writer, sheet_name='Calling_Recommendations', index=False)
        worksheet = writer.sheets['Calling_Recommendations']
        worksheet.column_dimensions['A'].width = 15
        worksheet.column_dimensions['B'].width = 40
        worksheet.column_dimensions['C'].width = 50
        worksheet.column_dimensions['D'].width = 20
        worksheet.column_dimensions['E'].width = 40
        
        # Sheet 7: Data Sample (for reference)
        if 'raw_data_sample' in results and not results['raw_data_sample'].empty:
            results['raw_data_sample'].to_excel(
                writer, 
                sheet_name='Data_Sample', 
                index=False
            )
        
        # Sheet 8: Performance Metrics
        if 'summary' in results:
            metrics_data = []
            summary = results['summary']
            
            key_metrics = [
                ('Total Conversion Opportunity', summary.get('International_Customers_Not_Using_P2P', 0) + 
                 summary.get('Domestic_Customers_Not_Using_P2P', 0)),
                ('International Conversion Potential %', summary.get('Conversion_Potential_%', 0)),
                ('P2P Market Penetration %', 
                 (summary.get('P2P_Users', 0) / max(summary.get('Total_Unique_Customers', 1), 1)) * 100),
                ('Service Utilization Index',
                 len(set(results.get('domestic_other_not_p2p', pd.DataFrame())['Services_Used'].sum().split('; '))) 
                 if not results.get('domestic_other_not_p2p', pd.DataFrame()).empty else 0)
            ]
            
            for metric, value in key_metrics:
                if isinstance(value, (int, float)):
                    formatted_value = f"{value:.2f}" if isinstance(value, float) else f"{value:,}"
                else:
                    formatted_value = str(value)
                
                metrics_data.append({
                    'Performance_Metric': metric,
                    'Current_Value': formatted_value,
                    'Target': 'TBD',
                    'Status': 'On Track' if (isinstance(value, (int, float)) and value > 0) else 'Needs Attention',
                    'Notes': 'Monitor weekly' if 'Conversion' in metric else 'Baseline established'
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df.to_excel(writer, sheet_name='Performance_Metrics', index=False)
            worksheet = writer.sheets['Performance_Metrics']
            worksheet.column_dimensions['A'].width = 35
            worksheet.column_dimensions['B'].width = 20
            worksheet.column_dimensions['C'].width = 15
            worksheet.column_dimensions['D'].width = 15
            worksheet.column_dimensions['E'].width = 30
    
    output.seek(0)
    return output.getvalue()

def create_detailed_reports(results):
    """Create additional detailed reports"""
    reports = {}
    
    if 'intl_not_p2p' in results and not results['intl_not_p2p'].empty:
        # Priority ranking for calling
        df_intl = results['intl_not_p2p'].copy()
        
        # Add priority based on recency and transaction volume
        if 'Days_Since_Last_Tx' in df_intl.columns and 'Total_Transaction_Amount' in df_intl.columns:
            def assign_priority(row):
                days = row['Days_Since_Last_Tx'] if isinstance(row['Days_Since_Last_Tx'], (int, float)) else 999
                amount = row['Total_Transaction_Amount'] if isinstance(row['Total_Transaction_Amount'], (int, float)) else 0
                
                if days <= 7 and amount > 1000:
                    return 'Priority 1: High Value & Recent'
                elif days <= 7:
                    return 'Priority 2: Recent User'
                elif amount > 1000:
                    return 'Priority 3: High Value'
                elif days <= 30:
                    return 'Priority 4: Active in Last Month'
                else:
                    return 'Priority 5: Historical User'
            
            df_intl['Calling_Priority'] = df_intl.apply(assign_priority, axis=1)
            reports['priority_calling_list'] = df_intl[['Customer_ID', 'Full_Name', 'Calling_Priority', 
                                                        'Last_Transaction_Date', 'Total_Transaction_Amount']]
    
    return reports
