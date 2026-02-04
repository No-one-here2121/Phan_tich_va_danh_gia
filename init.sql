-- Initialize the database for VNStock App
CREATE DATABASE IF NOT EXISTS vnstock_db;
USE vnstock_db;

-- Create users table (with password hashing support)
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create search history table
CREATE TABLE IF NOT EXISTS search_history (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_searches (user_id, searched_at DESC)
);

-- Create trading signals table (NEW)
CREATE TABLE IF NOT EXISTS trading_signals (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    signal_date DATE NOT NULL,
    
    -- Final signal
    final_signal VARCHAR(10) NOT NULL, -- 'BUY', 'SELL', 'HOLD'
    final_confidence FLOAT NOT NULL, -- 0-1
    
    -- Individual signals
    sma_signal VARCHAR(10),
    sma_confidence FLOAT,
    sma_price FLOAT,
    sma20 FLOAT,
    sma50 FLOAT,
    sma200 FLOAT,
    
    rsi_signal VARCHAR(10),
    rsi_confidence FLOAT,
    rsi_value FLOAT,
    
    macd_signal VARCHAR(10),
    macd_confidence FLOAT,
    macd_value FLOAT,
    signal_line FLOAT,
    histogram FLOAT,
    
    bb_signal VARCHAR(10),
    bb_confidence FLOAT,
    bb_upper FLOAT,
    bb_middle FLOAT,
    bb_lower FLOAT,
    
    volume_ratio FLOAT,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_signal (user_id, symbol, signal_date DESC),
    UNIQUE KEY unique_user_signal_date (user_id, symbol, signal_date)
);

-- Create trading recommendations table (NEW)
CREATE TABLE IF NOT EXISTS trading_recommendations (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    
    -- Recommendation details
    action VARCHAR(20) NOT NULL, -- 'BUY', 'SELL', 'HOLD', 'INCREASE_POSITION', 'REDUCE_POSITION'
    confidence_score FLOAT NOT NULL, -- 0-100
    reason TEXT,
    
    -- Price targets
    entry_price FLOAT,
    target_price FLOAT,
    stop_loss FLOAT,
    
    -- Status
    status VARCHAR(20) DEFAULT 'ACTIVE', -- 'ACTIVE', 'COMPLETED', 'CANCELLED'
    
    -- Performance tracking
    realized_price FLOAT,
    pnl FLOAT,
    pnl_percentage FLOAT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_recommendations (user_id, created_at DESC),
    INDEX idx_user_symbol_status (user_id, symbol, status)
);

-- Create backtest results table (NEW)
CREATE TABLE IF NOT EXISTS backtest_results (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    
    -- Backtest parameters
    strategy_name VARCHAR(100),
    start_date DATE,
    end_date DATE,
    
    -- Results
    total_trades INT,
    winning_trades INT,
    losing_trades INT,
    win_rate FLOAT,
    
    total_return FLOAT,
    annual_return FLOAT,
    max_drawdown FLOAT,
    sharpe_ratio FLOAT,
    
    -- Details
    result_data LONGTEXT, -- JSON format with detailed trade logs
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_backtest (user_id, created_at DESC)
);

-- Create watchlist table (BONUS)
CREATE TABLE IF NOT EXISTS watchlists (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    company_name VARCHAR(255),
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    UNIQUE KEY unique_user_watchlist (user_id, symbol),
    INDEX idx_user_watchlist (user_id)
);

-- Insert a test user with MD5 hashed password
-- MD5('admin123') = 0192023a7bbd73250516f069df18b500
INSERT IGNORE INTO users (username, password, email) 
VALUES ('admin', '0192023a7bbd73250516f069df18b500', 'admin@vnstock.local');

-- Insert another test user
-- MD5('testpass') = 5a105e8b9d40e1329780d62ea2265d8a
INSERT IGNORE INTO users (username, password, email) 
VALUES ('testuser', '5a105e8b9d40e1329780d62ea2265d8a', 'testuser@vnstock.local');

-- Show tables to confirm
SHOW TABLES;
