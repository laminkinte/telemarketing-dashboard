import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import io
import warnings
from typing import Dict, List, Tuple, Any
warnings.filterwarnings('ignore')

# Try to import plotly with fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not installed. Some visualizations will be disabled.")
    # Create mock functions to avoid errors
    class MockPlotly:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    px = MockPlotly()
    go = MockPlotly()

# Set page configuration
st.set_page_config(
    page_title="Advanced Transaction Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3B82F6;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .tab-header {
        font-size: 1.2rem;
        color: #1E3A8A;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .segmentation-box {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
        margin: 0.5rem 0;
    }
    .percentage-indicator {
        font-size: 1.2rem;
        font-weight: bold;
        color: #3B82F6;
    }
    .export-section {
        background-color: #f0fdf4;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #22c55e;
        margin: 1rem 0;
    }
    .summary-box {
        background-color: #fef3c7;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class TransactionAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.initialize_product_categories()
        
    def initialize_product_categories(self):
        """Initialize all product categories for analysis"""
        self.product_categories = {
            'p2p': [
                'Internal Wallet Transfer (P2P)', 'Internal Wallet Transfer', 
                'P2P Transfer', 'Wallet Transfer', 'P2P', 'Peer to Peer'
            ],
            'international': [
                'International Remittance', 'International Transfer', 
                'Remittance', 'International', 'Receive Remittance',
                'Cross Border', 'Foreign Transfer'
            ],
            'deposit': [
                'Deposit', 'Cash In', 'Deposit Customer', 
                'Deposit Agent', 'Bank Deposit', 'Fund Load'
            ],
            'withdrawal': [
                'Withdrawal', 'Scan To Withdraw Agent', 
                'Scan To Withdraw Customer', 'Cash Out', 
                'Withdraw', 'ATM Withdrawal', 'Cash Pickup'
            ],
            'bill_payment': [
                'Bill Payment', 'Airtime Topup', 'Utility Payment', 
                'Topup', 'Electricity Bill', 'Water Bill'
            ],
            'other_services': [
                'Merchant Payment', 'Ticket', 'Scan To Send', 
                'Loan', 'Investment', 'Insurance'
            ]
        }
    
    def get_customers_by_product(self, product_type: str) -> set:
        """Get unique customers for a product category"""
        products = self.product_categories.get(product_type, [])
        mask = self.df['Product Name'].str.contains('|'.join(products), case=False, na=False)
        return set(self.df[mask]['User Identifier'].dropna().unique())
    
    def get_intl_received_customers(self) -> set:
        """Get customers who received international remittance"""
        international_products = self.product_categories['international']
        
        if 'Direction' in self.df.columns:
            mask = (
                self.df['Product Name'].str.contains('|'.join(international_products), case=False, na=False) &
                self.df['Direction'].str.contains('In|Receive|Credit|Income', case=False, na=False)
            )
        else:
            mask = self.df['Product Name'].str.contains('|'.join(international_products), case=False, na=False)
        
        return set(self.df[mask]['User Identifier'].dropna().unique())

class FilterAnalyzer:
    """Class to handle all filter analyses"""
    
    def __init__(self, analyzer: TransactionAnalyzer):
        self.analyzer = analyzer
        self.df = analyzer.df
    
    # ========== FILTER 1: International but NOT P2P ==========
    def analyze_intl_not_p2p(self) -> Dict[str, Any]:
        """Customers who did International remittance but not P2P"""
        intl_customers = self.analyzer.get_customers_by_product('international')
        p2p_customers = self.analyzer.get_customers_by_product('p2p')
        
        target_customers = intl_customers - p2p_customers
        
        # Create detailed report
        report_data = []
        for cust_id in list(target_customers)[:1000]:  # Limit for performance
            cust_df = self.df[self.df['User Identifier'] == cust_id]
            if not cust_df.empty:
                # Get customer info
                name = self._get_customer_name(cust_df)
                intl_transactions = self._get_intl_transactions(cust_df)
                last_intl_date = intl_transactions['Created At'].max() if not intl_transactions.empty else None
                
                report_data.append({
                    'CUSTOMER_ID': cust_id,
                    'FULL_NAME': name,
                    'INTERNATIONAL_TRANSACTION_COUNT': len(intl_transactions),
                    'LAST_INTERNATIONAL_DATE': last_intl_date.strftime('%Y-%m-%d') if last_intl_date else 'N/A',
                    'TOTAL_TRANSACTIONS': len(cust_df),
                    'OTHER_SERVICES_USED': self._get_other_services(cust_df),
                    'DAYS_SINCE_LAST_INTL': (datetime.now() - last_intl_date).days if last_intl_date else 'N/A',
                    'CUSTOMER_SINCE': cust_df['Created At'].min().strftime('%Y-%m-%d')
                })
        
        return {
            'data': pd.DataFrame(report_data) if report_data else pd.DataFrame(),
            'summary': {
                'total_customers': len(target_customers),
                'intl_customers': len(intl_customers),
                'p2p_customers': len(p2p_customers),
                'description': 'Customers who performed International remittance but NEVER used P2P services'
            }
        }
    
    # ========== FILTER 2: P2P but NOT Deposit ==========
    def analyze_p2p_not_deposit(self) -> Dict[str, Any]:
        """Customers who did P2P but not Deposit"""
        p2p_customers = self.analyzer.get_customers_by_product('p2p')
        deposit_customers = self.analyzer.get_customers_by_product('deposit')
        
        target_customers = p2p_customers - deposit_customers
        
        report_data = []
        for cust_id in list(target_customers)[:1000]:
            cust_df = self.df[self.df['User Identifier'] == cust_id]
            if not cust_df.empty:
                name = self._get_customer_name(cust_df)
                p2p_transactions = self._get_p2p_transactions(cust_df)
                
                report_data.append({
                    'CUSTOMER_ID': cust_id,
                    'FULL_NAME': name,
                    'P2P_TRANSACTION_COUNT': len(p2p_transactions),
                    'LAST_P2P_DATE': p2p_transactions['Created At'].max().strftime('%Y-%m-%d') if not p2p_transactions.empty else 'N/A',
                    'TOTAL_TRANSACTIONS': len(cust_df),
                    'AVG_P2P_AMOUNT': p2p_transactions['Amount'].mean() if 'Amount' in p2p_transactions.columns else 'N/A',
                    'P2P_FREQUENCY_DAYS': self._calculate_transaction_frequency(p2p_transactions),
                    'CUSTOMER_SINCE': cust_df['Created At'].min().strftime('%Y-%m-%d')
                })
        
        return {
            'data': pd.DataFrame(report_data) if report_data else pd.DataFrame(),
            'summary': {
                'total_customers': len(target_customers),
                'p2p_customers': len(p2p_customers),
                'deposit_customers': len(deposit_customers),
                'description': 'Customers who used P2P services but NEVER made any deposits'
            }
        }
    
    # ========== FILTER 3: P2P AND Withdrawal ==========
    def analyze_p2p_and_withdrawal(self) -> Dict[str, Any]:
        """Customers who did P2P and withdrawal"""
        p2p_customers = self.analyzer.get_customers_by_product('p2p')
        withdrawal_customers = self.analyzer.get_customers_by_product('withdrawal')
        
        target_customers = p2p_customers & withdrawal_customers
        
        report_data = []
        for cust_id in list(target_customers)[:1000]:
            cust_df = self.df[self.df['User Identifier'] == cust_id]
            if not cust_df.empty:
                name = self._get_customer_name(cust_df)
                p2p_count = len(self._get_p2p_transactions(cust_df))
                withdrawal_count = len(self._get_withdrawal_transactions(cust_df))
                
                report_data.append({
                    'CUSTOMER_ID': cust_id,
                    'FULL_NAME': name,
                    'P2P_COUNT': p2p_count,
                    'WITHDRAWAL_COUNT': withdrawal_count,
                    'TOTAL_TRANSACTIONS': len(cust_df),
                    'P2P_TO_WITHDRAWAL_RATIO': f"{p2p_count/withdrawal_count:.2f}" if withdrawal_count > 0 else 'N/A',
                    'LAST_P2P_DATE': self._get_p2p_transactions(cust_df)['Created At'].max().strftime('%Y-%m-%d') if p2p_count > 0 else 'N/A',
                    'LAST_WITHDRAWAL_DATE': self._get_withdrawal_transactions(cust_df)['Created At'].max().strftime('%Y-%m-%d') if withdrawal_count > 0 else 'N/A',
                    'CUSTOMER_SINCE': cust_df['Created At'].min().strftime('%Y-%m-%d')
                })
        
        return {
            'data': pd.DataFrame(report_data) if report_data else pd.DataFrame(),
            'summary': {
                'total_customers': len(target_customers),
                'p2p_customers': len(p2p_customers),
                'withdrawal_customers': len(withdrawal_customers),
                'description': 'Customers who used BOTH P2P and Withdrawal services'
            }
        }
    
    # ========== FILTER 4: International AND P2P ==========
    def analyze_intl_and_p2p(self) -> Dict[str, Any]:
        """Customers who did international remittances and P2P"""
        intl_customers = self.analyzer.get_customers_by_product('international')
        p2p_customers = self.analyzer.get_customers_by_product('p2p')
        
        target_customers = intl_customers & p2p_customers
        
        report_data = []
        for cust_id in list(target_customers)[:1000]:
            cust_df = self.df[self.df['User Identifier'] == cust_id]
            if not cust_df.empty:
                name = self._get_customer_name(cust_df)
                intl_transactions = self._get_intl_transactions(cust_df)
                p2p_transactions = self._get_p2p_transactions(cust_df)
                
                report_data.append({
                    'CUSTOMER_ID': cust_id,
                    'FULL_NAME': name,
                    'INTERNATIONAL_COUNT': len(intl_transactions),
                    'P2P_COUNT': len(p2p_transactions),
                    'TOTAL_TRANSACTIONS': len(cust_df),
                    'LAST_INTERNATIONAL_DATE': intl_transactions['Created At'].max().strftime('%Y-%m-%d') if not intl_transactions.empty else 'N/A',
                    'LAST_P2P_DATE': p2p_transactions['Created At'].max().strftime('%Y-%m-%d') if not p2p_transactions.empty else 'N/A',
                    'INTERNATIONAL_TO_P2P_RATIO': f"{len(intl_transactions)/len(p2p_transactions):.2f}" if len(p2p_transactions) > 0 else 'N/A',
                    'CUSTOMER_SINCE': cust_df['Created At'].min().strftime('%Y-%m-%d')
                })
        
        return {
            'data': pd.DataFrame(report_data) if report_data else pd.DataFrame(),
            'summary': {
                'total_customers': len(target_customers),
                'intl_customers': len(intl_customers),
                'p2p_customers': len(p2p_customers),
                'description': 'Customers who used BOTH International remittance and P2P services'
            }
        }
    
    # ========== FILTER 5: Received International & Withdrew (Segmented) ==========
    def analyze_intl_received_withdrawal_segmented(self) -> Dict[str, Any]:
        """Customers who Received International Remittance and followed by withdrawal"""
        intl_received_customers = self.analyzer.get_intl_received_customers()
        
        detailed_report = []
        segmentation_data = {
            'segment_0_25': {'customers': [], 'count': 0, 'range': '≤25%'},
            'segment_26_50': {'customers': [], 'count': 0, 'range': '26-50%'},
            'segment_51_75': {'customers': [], 'count': 0, 'range': '51-75%'},
            'segment_76_100': {'customers': [], 'count': 0, 'range': '76-100%'},
            'segment_over_100': {'customers': [], 'count': 0, 'range': '>100%'}
        }
        
        for cust_id in list(intl_received_customers)[:1000]:
            cust_df = self.df[self.df['User Identifier'] == cust_id].sort_values('Created At')
            
            # Get international received transactions
            intl_received = self._get_intl_received_transactions(cust_df)
            if intl_received.empty:
                continue
            
            # Get withdrawals after first international receipt
            first_intl_date = intl_received['Created At'].min()
            withdrawals = self._get_withdrawals_after_date(cust_df, first_intl_date)
            
            if withdrawals.empty:
                continue
            
            # Calculate amounts
            total_received = intl_received['Amount'].sum() if 'Amount' in intl_received.columns else len(intl_received)
            total_withdrawn = withdrawals['Amount'].sum() if 'Amount' in withdrawals.columns else len(withdrawals)
            
            withdrawal_percentage = (total_withdrawn / total_received * 100) if total_received > 0 else 0
            
            # Segment customer
            if withdrawal_percentage <= 25:
                segment = 'segment_0_25'
            elif withdrawal_percentage <= 50:
                segment = 'segment_26_50'
            elif withdrawal_percentage <= 75:
                segment = 'segment_51_75'
            elif withdrawal_percentage <= 100:
                segment = 'segment_76_100'
            else:
                segment = 'segment_over_100'
            
            segmentation_data[segment]['customers'].append(cust_id)
            segmentation_data[segment]['count'] += 1
            
            detailed_report.append({
                'CUSTOMER_ID': cust_id,
                'FULL_NAME': self._get_customer_name(cust_df),
                'TOTAL_RECEIVED_AMOUNT': total_received,
                'TOTAL_WITHDRAWN_AMOUNT': total_withdrawn,
                'WITHDRAWAL_PERCENTAGE': f"{withdrawal_percentage:.2f}%",
                'WITHDRAWAL_SEGMENT': segmentation_data[segment]['range'],
                'INTERNATIONAL_COUNT': len(intl_received),
                'WITHDRAWAL_COUNT': len(withdrawals),
                'FIRST_RECEIVED_DATE': first_intl_date.strftime('%Y-%m-%d'),
                'LAST_WITHDRAWAL_DATE': withdrawals['Created At'].max().strftime('%Y-%m-%d'),
                'DAYS_BETWEEN_FIRST_RECEIVED_LAST_WITHDRAWAL': (withdrawals['Created At'].max() - first_intl_date).days
            })
        
        return {
            'data': pd.DataFrame(detailed_report) if detailed_report else pd.DataFrame(),
            'segmentation': segmentation_data,
            'summary': {
                'total_customers': len(detailed_report),
                'intl_received_customers': len(intl_received_customers),
                'description': 'Customers who Received International Remittance and made withdrawals (segmented by withdrawal percentage)'
            }
        }
    
    # ========== FILTER 6: Received International, NO Withdrawal, Uses Other Services ==========
    def analyze_intl_received_no_withdrawal_other(self) -> Dict[str, Any]:
        """Customers who Received International Remittance did not withdraw and use other services"""
        intl_received_customers = self.analyzer.get_intl_received_customers()
        withdrawal_customers = self.analyzer.get_customers_by_product('withdrawal')
        
        # Customers who received international but didn't withdraw
        target_customers = intl_received_customers - withdrawal_customers
        
        # Filter only those who used other services
        other_service_customers = self.analyzer.get_customers_by_product('bill_payment').union(
            self.analyzer.get_customers_by_product('other_services')
        )
        
        final_customers = target_customers & other_service_customers
        
        report_data = []
        for cust_id in list(final_customers)[:1000]:
            cust_df = self.df[self.df['User Identifier'] == cust_id]
            if not cust_df.empty:
                name = self._get_customer_name(cust_df)
                intl_received = self._get_intl_received_transactions(cust_df)
                
                # Get other services used
                other_services = self._get_other_services_detailed(cust_df)
                
                report_data.append({
                    'CUSTOMER_ID': cust_id,
                    'FULL_NAME': name,
                    'TOTAL_RECEIVED_AMOUNT': intl_received['Amount'].sum() if 'Amount' in intl_received.columns else len(intl_received),
                    'INTERNATIONAL_COUNT': len(intl_received),
                    'OTHER_SERVICES_COUNT': len(other_services),
                    'OTHER_SERVICES_USED': ', '.join(other_services[:5]),  # Top 5 services
                    'LAST_INTERNATIONAL_DATE': intl_received['Created At'].max().strftime('%Y-%m-%d') if not intl_received.empty else 'N/A',
                    'LAST_OTHER_SERVICE_DATE': self._get_last_other_service_date(cust_df),
                    'CUSTOMER_SINCE': cust_df['Created At'].min().strftime('%Y-%m-%d'),
                    'ACTIVITY_SCORE': self._calculate_activity_score(cust_df)
                })
        
        return {
            'data': pd.DataFrame(report_data) if report_data else pd.DataFrame(),
            'summary': {
                'total_customers': len(final_customers),
                'intl_received_customers': len(intl_received_customers),
                'description': 'Customers who Received International Remittance, did NOT withdraw, but used other services'
            }
        }
    
    # ========== FILTER 7: Received International & ONLY Withdrew ==========
    def analyze_intl_received_only_withdrawal(self) -> Dict[str, Any]:
        """Customers who Received International Remittance and only withdraw and did not use any other services"""
        intl_received_customers = self.analyzer.get_intl_received_customers()
        withdrawal_customers = self.analyzer.get_customers_by_product('withdrawal')
        
        # Customers who received international and withdrew
        target_customers = intl_received_customers & withdrawal_customers
        
        # Filter out customers who used any other services
        other_service_customers = (
            self.analyzer.get_customers_by_product('p2p') |
            self.analyzer.get_customers_by_product('deposit') |
            self.analyzer.get_customers_by_product('bill_payment') |
            self.analyzer.get_customers_by_product('other_services')
        )
        
        final_customers = target_customers - other_service_customers
        
        report_data = []
        for cust_id in list(final_customers)[:1000]:
            cust_df = self.df[self.df['User Identifier'] == cust_id]
            if not cust_df.empty:
                name = self._get_customer_name(cust_df)
                intl_received = self._get_intl_received_transactions(cust_df)
                withdrawals = self._get_withdrawal_transactions(cust_df)
                
                # Calculate withdrawal efficiency
                total_received = intl_received['Amount'].sum() if 'Amount' in intl_received.columns else len(intl_received)
                total_withdrawn = withdrawals['Amount'].sum() if 'Amount' in withdrawals.columns else len(withdrawals)
                
                report_data.append({
                    'CUSTOMER_ID': cust_id,
                    'FULL_NAME': name,
                    'TOTAL_RECEIVED_AMOUNT': total_received,
                    'TOTAL_WITHDRAWN_AMOUNT': total_withdrawn,
                    'WITHDRAWAL_EFFICIENCY': f"{(total_withdrawn/total_received*100):.2f}%" if total_received > 0 else 'N/A',
                    'INTERNATIONAL_COUNT': len(intl_received),
                    'WITHDRAWAL_COUNT': len(withdrawals),
                    'AVG_TIME_TO_WITHDRAWAL_DAYS': self._calculate_avg_withdrawal_time(cust_df),
                    'FIRST_RECEIVED_DATE': intl_received['Created At'].min().strftime('%Y-%m-%d') if not intl_received.empty else 'N/A',
                    'LAST_WITHDRAWAL_DATE': withdrawals['Created At'].max().strftime('%Y-%m-%d') if not withdrawals.empty else 'N/A',
                    'CUSTOMER_SINCE': cust_df['Created At'].min().strftime('%Y-%m-%d'),
                    'WITHDRAWAL_FREQUENCY': self._calculate_withdrawal_frequency(cust_df)
                })
        
        return {
            'data': pd.DataFrame(report_data) if report_data else pd.DataFrame(),
            'summary': {
                'total_customers': len(final_customers),
                'intl_received_customers': len(intl_received_customers),
                'description': 'Customers who Received International Remittance and ONLY withdrew (no other services)'
            }
        }
    
    # ========== HELPER METHODS ==========
    def _get_customer_name(self, cust_df: pd.DataFrame) -> str:
        """Extract customer name from dataframe"""
        name_records = cust_df[cust_df['Full Name'].notna() & (cust_df['Full Name'] != 'nan')]
        if not name_records.empty:
            return name_records.iloc[0]['Full Name']
        return 'Name not available'
    
    def _get_intl_transactions(self, cust_df: pd.DataFrame) -> pd.DataFrame:
        """Get international transactions for a customer"""
        products = self.analyzer.product_categories['international']
        mask = cust_df['Product Name'].str.contains('|'.join(products), case=False, na=False)
        return cust_df[mask]
    
    def _get_intl_received_transactions(self, cust_df: pd.DataFrame) -> pd.DataFrame:
        """Get international received transactions for a customer"""
        products = self.analyzer.product_categories['international']
        
        if 'Direction' in cust_df.columns:
            mask = (
                cust_df['Product Name'].str.contains('|'.join(products), case=False, na=False) &
                cust_df['Direction'].str.contains('In|Receive|Credit|Income', case=False, na=False)
            )
        else:
            mask = cust_df['Product Name'].str.contains('|'.join(products), case=False, na=False)
        
        return cust_df[mask]
    
    def _get_p2p_transactions(self, cust_df: pd.DataFrame) -> pd.DataFrame:
        """Get P2P transactions for a customer"""
        products = self.analyzer.product_categories['p2p']
        mask = cust_df['Product Name'].str.contains('|'.join(products), case=False, na=False)
        return cust_df[mask]
    
    def _get_withdrawal_transactions(self, cust_df: pd.DataFrame) -> pd.DataFrame:
        """Get withdrawal transactions for a customer"""
        products = self.analyzer.product_categories['withdrawal']
        mask = cust_df['Product Name'].str.contains('|'.join(products), case=False, na=False)
        return cust_df[mask]
    
    def _get_withdrawals_after_date(self, cust_df: pd.DataFrame, date: datetime) -> pd.DataFrame:
        """Get withdrawal transactions after a specific date"""
        withdrawals = self._get_withdrawal_transactions(cust_df)
        return withdrawals[withdrawals['Created At'] > date]
    
    def _get_other_services(self, cust_df: pd.DataFrame) -> str:
        """Get other services used by customer"""
        other_services = []
        for service_type in ['bill_payment', 'other_services']:
            products = self.analyzer.product_categories[service_type]
            mask = cust_df['Product Name'].str.contains('|'.join(products), case=False, na=False)
            if mask.any():
                services = cust_df[mask]['Product Name'].unique()
                other_services.extend([str(s) for s in services])
        
        return ', '.join(other_services[:3]) if other_services else 'None'
    
    def _get_other_services_detailed(self, cust_df: pd.DataFrame) -> List[str]:
        """Get detailed list of other services used"""
        other_services = []
        for service_type in ['bill_payment', 'other_services', 'p2p', 'deposit']:
            products = self.analyzer.product_categories.get(service_type, [])
            mask = cust_df['Product Name'].str.contains('|'.join(products), case=False, na=False)
            if mask.any():
                services = cust_df[mask]['Product Name'].unique()
                other_services.extend([str(s) for s in services])
        
        return list(set(other_services))
    
    def _get_last_other_service_date(self, cust_df: pd.DataFrame) -> str:
        """Get date of last other service used"""
        other_dates = []
        for service_type in ['bill_payment', 'other_services', 'p2p', 'deposit']:
            products = self.analyzer.product_categories.get(service_type, [])
            mask = cust_df['Product Name'].str.contains('|'.join(products), case=False, na=False)
            if mask.any():
                last_date = cust_df[mask]['Created At'].max()
                if pd.notna(last_date):
                    other_dates.append(last_date)
        
        if other_dates:
            return max(other_dates).strftime('%Y-%m-%d')
        return 'N/A'
    
    def _calculate_transaction_frequency(self, trans_df: pd.DataFrame) -> str:
        """Calculate average days between transactions"""
        if len(trans_df) < 2:
            return 'N/A'
        
        trans_df = trans_df.sort_values('Created At')
        time_diffs = trans_df['Created At'].diff().dt.total_seconds() / (24 * 3600)
        avg_days = time_diffs.mean()
        return f"{avg_days:.1f} days"
    
    def _calculate_activity_score(self, cust_df: pd.DataFrame) -> str:
        """Calculate customer activity score"""
        total_transactions = len(cust_df)
        days_active = (cust_df['Created At'].max() - cust_df['Created At'].min()).days + 1
        transaction_per_day = total_transactions / days_active if days_active > 0 else 0
        
        # Simple scoring
        if transaction_per_day > 1:
            return "High"
        elif transaction_per_day > 0.5:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_avg_withdrawal_time(self, cust_df: pd.DataFrame) -> str:
        """Calculate average time between receiving and withdrawing"""
        intl_received = self._get_intl_received_transactions(cust_df)
        withdrawals = self._get_withdrawal_transactions(cust_df)
        
        if intl_received.empty or withdrawals.empty:
            return 'N/A'
        
        times = []
        for _, intl_row in intl_received.iterrows():
            intl_date = intl_row['Created At']
            later_withdrawals = withdrawals[withdrawals['Created At'] > intl_date]
            if not later_withdrawals.empty:
                first_withdrawal = later_withdrawals.iloc[0]
                time_diff = (first_withdrawal['Created At'] - intl_date).days
                times.append(time_diff)
        
        if times:
            avg_time = sum(times) / len(times)
            return f"{avg_time:.1f} days"
        return 'N/A'
    
    def _calculate_withdrawal_frequency(self, cust_df: pd.DataFrame) -> str:
        """Calculate withdrawal frequency"""
        withdrawals = self._get_withdrawal_transactions(cust_df)
        if len(withdrawals) < 2:
            return 'N/A'
        
        withdrawals = withdrawals.sort_values('Created At')
        time_diffs = withdrawals['Created At'].diff().dt.total_seconds() / (24 * 3600)
        avg_days = time_diffs.mean()
        return f"Every {avg_days:.1f} days"

def load_data(uploaded_file):
    """Load and clean the transaction data"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
        
        st.success(f"✅ Successfully loaded {len(df):,} transactions")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Check required columns
        required_columns = ['User Identifier', 'Product Name', 'Created At']
        optional_columns = ['Service Name', 'Entity Name', 'Full Name', 'Amount', 'Transaction Type', 'Direction']
        
        # Check for minimum required columns
        missing_required = [col for col in required_columns if col not in df.columns]
        if missing_required:
            st.error(f"Missing required columns: {missing_required}")
            st.info(f"Available columns: {list(df.columns)}")
            return None
        
        # Add missing optional columns with default values
        for col in optional_columns:
            if col not in df.columns:
                if col == 'Amount':
                    df[col] = 1.0  # Default amount if not available
                elif col == 'Direction':
                    df[col] = 'Credit'  # Default direction
                else:
                    df[col] = ''
                st.warning(f"Note: Column '{col}' not found. Using default values.")
        
        # Data cleaning
        for col in ['Product Name', 'Service Name', 'Entity Name', 'Full Name', 'Transaction Type', 'Direction']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
        
        df['User Identifier'] = pd.to_numeric(df['User Identifier'], errors='coerce')
        
        # Parse date column
        date_formats = ['%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', 
                       '%d/%m/%Y %H:%M:%S', '%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']
        
        for fmt in date_formats:
            try:
                df['Created At'] = pd.to_datetime(df['Created At'], format=fmt, errors='coerce')
                if df['Created At'].notna().any():
                    break
            except:
                continue
        
        # If still not parsed, try generic parsing
        if df['Created At'].isna().all():
            df['Created At'] = pd.to_datetime(df['Created At'], errors='coerce')
        
        # Add derived columns
        df['Date'] = df['Created At'].dt.date
        df['Day'] = df['Created At'].dt.day_name()
        df['Hour'] = df['Created At'].dt.hour
        df['Month'] = df['Created At'].dt.month
        df['Year'] = df['Created At'].dt.year
        
        return df
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

def create_comprehensive_excel(all_results: Dict[str, Any]) -> io.BytesIO:
    """Create comprehensive Excel workbook with all analyses"""
    output = io.BytesIO()
    
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Summary Sheet
            summary_data = []
            for filter_name, result in all_results.items():
                if result and 'summary' in result:
                    summary = result['summary']
                    summary_data.append({
                        'Filter Type': filter_name.replace('_', ' ').title(),
                        'Total Customers': summary.get('total_customers', 0),
                        'Description': summary.get('description', ''),
                        'Reference Populations': self._format_reference_populations(summary)
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='00_SUMMARY', index=False)
            
            # Individual Analysis Sheets
            sheet_order = [
                ('intl_not_p2p', '01_INTL_NOT_P2P'),
                ('p2p_not_deposit', '02_P2P_NOT_DEPOSIT'),
                ('p2p_and_withdrawal', '03_P2P_AND_WITHDRAWAL'),
                ('intl_and_p2p', '04_INTL_AND_P2P'),
                ('intl_received_withdrawal_segmented', '05_INTL_RECEIVED_WITHDRAWAL_SEGMENTED'),
                ('intl_received_no_withdrawal_other', '06_INTL_RECEIVED_NO_WITHDRAWAL_OTHER'),
                ('intl_received_only_withdrawal', '07_INTL_RECEIVED_ONLY_WITHDRAWAL')
            ]
            
            for filter_key, sheet_name in sheet_order:
                if filter_key in all_results and all_results[filter_key]:
                    result = all_results[filter_key]
                    
                    # Write main data
                    if not result['data'].empty:
                        result['data'].to_excel(writer, sheet_name=sheet_name, index=False)
                    
                    # Write segmentation data if available
                    if 'segmentation' in result:
                        seg_data = result['segmentation']
                        seg_df = pd.DataFrame([
                            {
                                'Segment Range': seg_info['range'],
                                'Customer Count': seg_info['count'],
                                'Percentage': f"{(seg_info['count']/result['summary']['total_customers']*100):.1f}%" if result['summary']['total_customers'] > 0 else '0%'
                            }
                            for seg_key, seg_info in seg_data.items()
                        ])
                        seg_df.to_excel(writer, sheet_name=f'{sheet_name}_SEGMENTATION', index=False)
            
            # Global Statistics Sheet
            global_stats = pd.DataFrame({
                'Statistic': [
                    'Total Transactions',
                    'Unique Customers',
                    'Date Range Start',
                    'Date Range End',
                    'Average Transactions per Customer',
                    'Most Active Product',
                    'Export Timestamp'
                ],
                'Value': [
                    len(all_results.get('raw_data', pd.DataFrame())),
                    all_results.get('raw_data', pd.DataFrame())['User Identifier'].nunique(),
                    all_results.get('raw_data', pd.DataFrame())['Created At'].min().strftime('%Y-%m-%d') if not all_results.get('raw_data', pd.DataFrame()).empty else 'N/A',
                    all_results.get('raw_data', pd.DataFrame())['Created At'].max().strftime('%Y-%m-%d') if not all_results.get('raw_data', pd.DataFrame()).empty else 'N/A',
                    f"{len(all_results.get('raw_data', pd.DataFrame()))/all_results.get('raw_data', pd.DataFrame())['User Identifier'].nunique():.2f}" if not all_results.get('raw_data', pd.DataFrame()).empty and all_results.get('raw_data', pd.DataFrame())['User Identifier'].nunique() > 0 else 'N/A',
                    all_results.get('raw_data', pd.DataFrame())['Product Name'].mode()[0] if not all_results.get('raw_data', pd.DataFrame()).empty and not all_results.get('raw_data', pd.DataFrame())['Product Name'].mode().empty else 'N/A',
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ]
            })
            global_stats.to_excel(writer, sheet_name='GLOBAL_STATS', index=False)
        
        output.seek(0)
        return output
    
    except Exception as e:
        st.error(f"Error creating Excel file: {str(e)}")
        return None

def display_segmentation_results(segmentation_data: Dict):
    """Display segmentation results"""
    if not segmentation_data:
        return
    
    st.markdown("### 📊 Withdrawal Percentage Segmentation")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    colors = ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6']
    
    for idx, (seg_key, seg_info) in enumerate(segmentation_data.items()):
        col = [col1, col2, col3, col4, col5][idx]
        with col:
            st.markdown(f'<div class="segmentation-box">', unsafe_allow_html=True)
            st.markdown(f'<div class="percentage-indicator" style="color: {colors[idx]};">{seg_info["range"]}</div>', unsafe_allow_html=True)
            st.metric("Customers", seg_info['count'])
            st.markdown(f'</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit app"""
    st.markdown('<h1 class="main-header">📊 Advanced Transaction Analysis Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Transaction Data",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your transaction file (CSV or Excel)"
        )
        
        if uploaded_file is not None:
            # Load data
            with st.spinner("Loading data..."):
                df = load_data(uploaded_file)
            
            if df is not None:
                # Date range filter
                st.subheader("📅 Date Range Filter")
                min_date = df['Created At'].min().date() if df['Created At'].notna().any() else datetime.now().date() - timedelta(days=30)
                max_date = df['Created At'].max().date() if df['Created At'].notna().any() else datetime.now().date()
                
                start_date = st.date_input(
                    "Start Date",
                    value=min_date,
                    min_value=min_date,
                    max_value=max_date
                )
                
                end_date = st.date_input(
                    "End Date",
                    value=max_date,
                    min_value=min_date,
                    max_value=max_date
                )
                
                # Apply date filter
                if start_date and end_date:
                    df_filtered = df[
                        (df['Created At'] >= pd.Timestamp(start_date)) & 
                        (df['Created At'] <= pd.Timestamp(end_date + timedelta(days=1)))
                    ].copy()
                else:
                    df_filtered = df.copy()
                
                st.info(f"📊 Filtered to {len(df_filtered):,} transactions")
                
                # Initialize analyzer
                analyzer = TransactionAnalyzer(df_filtered)
                filter_analyzer = FilterAnalyzer(analyzer)
                
                # Run analysis button
                if st.button("🚀 Run All Analyses", type="primary", use_container_width=True):
                    st.session_state['df'] = df_filtered
                    st.session_state['analyzer'] = analyzer
                    st.session_state['filter_analyzer'] = filter_analyzer
                    st.session_state['analysis_ready'] = True
                
                # Data preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df.head(20), use_container_width=True)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Rows", f"{len(df):,}")
                    with col2:
                        st.metric("Unique Customers", f"{df['User Identifier'].nunique():,}")
        else:
            st.info("👈 Please upload a transaction file to begin analysis")
            st.session_state['analysis_ready'] = False
    
    # Main content area
    if 'analysis_ready' in st.session_state and st.session_state['analysis_ready']:
        df_filtered = st.session_state['df']
        filter_analyzer = st.session_state['filter_analyzer']
        
        # Create tabs for each filter
        tab_names = [
            "1️⃣ International Not P2P",
            "2️⃣ P2P Not Deposit",
            "3️⃣ P2P & Withdrawal",
            "4️⃣ International & P2P",
            "5️⃣ Received & Withdrew (Segmented)",
            "6️⃣ Received, No Withdrawal, Other Services",
            "7️⃣ Received & Only Withdrew",
            "📊 Export All"
        ]
        
        tabs = st.tabs(tab_names)
        
        all_results = {}
        
        # Tab 1: International Not P2P
        with tabs[0]:
            st.markdown('<h2 class="tab-header">🎯 International Customers NOT Using P2P</h2>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing international customers without P2P..."):
                result = filter_analyzer.analyze_intl_not_p2p()
                all_results['intl_not_p2p'] = result
            
            if not result['data'].empty:
                # Display summary
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Target Customers", result['summary']['total_customers'])
                with col2:
                    st.metric("Total International", result['summary']['intl_customers'])
                with col3:
                    st.metric("Total P2P Users", result['summary']['p2p_customers'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display data
                st.dataframe(result['data'], use_container_width=True)
                st.info(f"📈 Found {len(result['data'])} customers who use international services but not P2P")
            else:
                st.warning("No customers found matching this criteria")
        
        # Tab 2: P2P Not Deposit
        with tabs[1]:
            st.markdown('<h2 class="tab-header">💸 P2P Customers NOT Making Deposits</h2>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing P2P customers without deposits..."):
                result = filter_analyzer.analyze_p2p_not_deposit()
                all_results['p2p_not_deposit'] = result
            
            if not result['data'].empty:
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Target Customers", result['summary']['total_customers'])
                with col2:
                    st.metric("Total P2P Users", result['summary']['p2p_customers'])
                with col3:
                    st.metric("Total Deposit Users", result['summary']['deposit_customers'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.dataframe(result['data'], use_container_width=True)
                st.info(f"📈 Found {len(result['data'])} customers who use P2P but don't make deposits")
            else:
                st.warning("No customers found matching this criteria")
        
        # Tab 3: P2P & Withdrawal
        with tabs[2]:
            st.markdown('<h2 class="tab-header">💳 P2P & Withdrawal Users</h2>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing P2P and withdrawal users..."):
                result = filter_analyzer.analyze_p2p_and_withdrawal()
                all_results['p2p_and_withdrawal'] = result
            
            if not result['data'].empty:
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Target Customers", result['summary']['total_customers'])
                with col2:
                    st.metric("Total P2P Users", result['summary']['p2p_customers'])
                with col3:
                    st.metric("Total Withdrawal Users", result['summary']['withdrawal_customers'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.dataframe(result['data'], use_container_width=True)
                st.info(f"📈 Found {len(result['data'])} customers who use both P2P and withdrawal services")
            else:
                st.warning("No customers found matching this criteria")
        
        # Tab 4: International & P2P
        with tabs[3]:
            st.markdown('<h2 class="tab-header">🌍 International & P2P Users</h2>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing international and P2P users..."):
                result = filter_analyzer.analyze_intl_and_p2p()
                all_results['intl_and_p2p'] = result
            
            if not result['data'].empty:
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Target Customers", result['summary']['total_customers'])
                with col2:
                    st.metric("Total International", result['summary']['intl_customers'])
                with col3:
                    st.metric("Total P2P Users", result['summary']['p2p_customers'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.dataframe(result['data'], use_container_width=True)
                st.info(f"📈 Found {len(result['data'])} customers who use both international and P2P services")
            else:
                st.warning("No customers found matching this criteria")
        
        # Tab 5: Received & Withdrew (Segmented)
        with tabs[4]:
            st.markdown('<h2 class="tab-header">💵 Received International & Withdrew (Segmented)</h2>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing international recipients with withdrawals..."):
                result = filter_analyzer.analyze_intl_received_withdrawal_segmented()
                all_results['intl_received_withdrawal_segmented'] = result
            
            if not result['data'].empty:
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Target Customers", result['summary']['total_customers'])
                with col2:
                    st.metric("Total International Recipients", result['summary']['intl_received_customers'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display segmentation
                if 'segmentation' in result:
                    display_segmentation_results(result['segmentation'])
                
                # Display data
                st.dataframe(result['data'], use_container_width=True)
                st.info(f"📈 Found {len(result['data'])} customers who received international and made withdrawals")
            else:
                st.warning("No customers found matching this criteria")
        
        # Tab 6: Received, No Withdrawal, Other Services
        with tabs[5]:
            st.markdown('<h2 class="tab-header">💰 Received, No Withdrawal, Other Services</h2>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing international recipients without withdrawals..."):
                result = filter_analyzer.analyze_intl_received_no_withdrawal_other()
                all_results['intl_received_no_withdrawal_other'] = result
            
            if not result['data'].empty:
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Target Customers", result['summary']['total_customers'])
                with col2:
                    st.metric("Total International Recipients", result['summary']['intl_received_customers'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.dataframe(result['data'], use_container_width=True)
                st.info(f"📈 Found {len(result['data'])} customers who received international, didn't withdraw, but used other services")
            else:
                st.warning("No customers found matching this criteria")
        
        # Tab 7: Received & Only Withdrew
        with tabs[6]:
            st.markdown('<h2 class="tab-header">💳 Received & Only Withdrew</h2>', unsafe_allow_html=True)
            
            with st.spinner("Analyzing international recipients who only withdrew..."):
                result = filter_analyzer.analyze_intl_received_only_withdrawal()
                all_results['intl_received_only_withdrawal'] = result
            
            if not result['data'].empty:
                st.markdown('<div class="summary-box">', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Target Customers", result['summary']['total_customers'])
                with col2:
                    st.metric("Total International Recipients", result['summary']['intl_received_customers'])
                st.markdown('</div>', unsafe_allow_html=True)
                
                st.dataframe(result['data'], use_container_width=True)
                st.info(f"📈 Found {len(result['data'])} customers who received international and only made withdrawals")
            else:
                st.warning("No customers found matching this criteria")
        
        # Tab 8: Export All
        with tabs[7]:
            st.markdown('<h2 class="tab-header">📥 Export All Analyses</h2>', unsafe_allow_html=True)
            
            st.markdown('<div class="export-section">', unsafe_allow_html=True)
            st.markdown("### 📊 Comprehensive Export")
            
            # Add raw data to results for export
            all_results['raw_data'] = df_filtered
            
            # Calculate total unique customers across all filters
            total_customers_covered = 0
            unique_customers = set()
            
            for filter_key, result in all_results.items():
                if filter_key != 'raw_data' and result and 'data' in result:
                    if not result['data'].empty and 'CUSTOMER_ID' in result['data'].columns:
                        customers = set(result['data']['CUSTOMER_ID'].unique())
                        unique_customers.update(customers)
                        total_customers_covered += len(customers)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Analyses", 7)
            with col2:
                st.metric("Total Customers Covered", len(unique_customers))
            with col3:
                st.metric("Total Rows to Export", sum([len(result['data']) for filter_key, result in all_results.items() if filter_key != 'raw_data']))
            
            st.markdown("### 📋 Export Contents")
            
            export_contents = [
                "✅ 00_SUMMARY - Overview of all analyses",
                "✅ 01_INTL_NOT_P2P - International customers without P2P",
                "✅ 02_P2P_NOT_DEPOSIT - P2P customers without deposits",
                "✅ 03_P2P_AND_WITHDRAWAL - Customers using both services",
                "✅ 04_INTL_AND_P2P - Customers using both services",
                "✅ 05_INTL_RECEIVED_WITHDRAWAL_SEGMENTED - Withdrawal percentage segmentation",
                "✅ 05_INTL_RECEIVED_WITHDRAWAL_SEGMENTED_SEGMENTATION - Segmentation details",
                "✅ 06_INTL_RECEIVED_NO_WITHDRAWAL_OTHER - Recipients using other services",
                "✅ 07_INTL_RECEIVED_ONLY_WITHDRAWAL - Recipients only withdrawing",
                "✅ GLOBAL_STATS - Overall statistics"
            ]
            
            for item in export_contents:
                st.write(item)
            
            # Export button
            if st.button("📥 Generate & Download Complete Excel Report", type="primary", use_container_width=True):
                with st.spinner("Creating comprehensive Excel report..."):
                    excel_data = create_comprehensive_excel(all_results)
                    
                    if excel_data:
                        # Create download button
                        st.download_button(
                            label="⬇️ Download Excel Report",
                            data=excel_data,
                            file_name=f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                        
                        st.success("✅ Report ready for download!")
                    else:
                        st.error("Failed to create Excel report")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display quick statistics
            st.markdown("### 📈 Quick Statistics")
            quick_stats = pd.DataFrame({
                'Analysis': [
                    'International Not P2P',
                    'P2P Not Deposit',
                    'P2P & Withdrawal',
                    'International & P2P',
                    'Received & Withdrew',
                    'Received, No Withdrawal',
                    'Received & Only Withdrew'
                ],
                'Customer Count': [
                    len(all_results.get('intl_not_p2p', {}).get('data', pd.DataFrame())),
                    len(all_results.get('p2p_not_deposit', {}).get('data', pd.DataFrame())),
                    len(all_results.get('p2p_and_withdrawal', {}).get('data', pd.DataFrame())),
                    len(all_results.get('intl_and_p2p', {}).get('data', pd.DataFrame())),
                    len(all_results.get('intl_received_withdrawal_segmented', {}).get('data', pd.DataFrame())),
                    len(all_results.get('intl_received_no_withdrawal_other', {}).get('data', pd.DataFrame())),
                    len(all_results.get('intl_received_only_withdrawal', {}).get('data', pd.DataFrame()))
                ]
            })
            
            st.dataframe(quick_stats, use_container_width=True)

if __name__ == "__main__":
    main()
