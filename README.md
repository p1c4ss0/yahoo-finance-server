# F.A.S.T. Trading Bot - Yahoo Finance Server (Fork)

[![Trading Bot](https://img.shields.io/badge/F.A.S.T-Trading%20Bot-blue)](https://github.com/p1c4ss0/trading-bot-yahoo-finance-server)
[![Original](https://img.shields.io/badge/Original-AgentX--ai-purple)](https://github.com/AgentX-ai/yahoo-finance-server)

📈 **Enhanced Yahoo Finance MCP server optimized for the F.A.S.T. Trading Bot** - includes comprehensive stock market data, Indian market optimizations, trading-specific analytics, and filtered news for better trading decisions.

## ✨ Features

### Original Features
- 📊 **Stock Data** - Get comprehensive ticker information including company details, financials, and trading metrics
- 📰 **News Feed** - Fetch recent news articles related to specific stock symbols
- 🔍 **Search** - Find stocks, ETFs, and other financial instruments with news results
- 🏆 **Top Entities** - Get top performing companies, ETFs, and mutual funds by sector
- 📈 **Price History** - Fetch historical price data with customizable periods and intervals
- ⚡ **Options Chain** - Get option chain data for stocks including calls and puts
- 💰 **Earnings** - Access earnings data including upcoming earnings dates
- 🌐 **Proxy Support** - Works with HTTP/HTTPS/SOCKS proxies
- 🚀 **Fast & Async** - Non-blocking operations using asyncio
- 🔧 **Easy Integration** - Standard MCP protocol for AI assistants

### 🎯 Trading Bot Enhancements
- 🇮🇳 **Indian Market Status** - Real-time NSE market hours and status monitoring
- 📊 **NSE Sector Performance** - Sector momentum analysis using ETFs with trading scores
- 📈 **Enhanced Ticker Info** - Trading-specific metrics including SMA comparison, volume surges, volatility indicators
- 🎯 **Filtered Trading News** - AI-powered news filtering for trading relevance with scoring
- ⚡ **Volume Anomaly Detection** - Automated detection of unusual volume spikes
- 📊 **Technical Indicators** - Built-in SMA, ATR, and momentum calculations
- 🚨 **Risk Metrics** - Volatility and risk assessment for better position sizing

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- pip or uvx package manager
- (Optional) Proxy server for better reliability

### Installation

#### Using uvx (Recommended)

```bash
# Install with namespace
uvx yahoo-finance-server
```

#### Using pip

```bash
pip install yahoo-finance-server
```

#### From source

```bash
git clone https://github.com/AgentX-ai/AgentX-mcp-servers.git
cd AgentX-mcp-servers/yahoo_finance_server
pip install -e .
```

### Configuration

#### Proxy Setup (Recommended)

For better reliability and to avoid rate limiting, set up a proxy:

```bash
# HTTP/HTTPS proxy
export PROXY_URL="http://proxy.example.com:8080"

# SOCKS proxy with auth
export PROXY_URL="socks5://user:pass@127.0.0.1:1080/"
```

#### Running the Server

```bash
# Basic run
yahoo-finance-server

# Run with proxy
PROXY_URL="http://127.0.0.1:7890" yahoo-finance-server
```

## 🛠️ API Reference

### Available Tools

#### 1. **get-ticker-info**

Get comprehensive stock information including company details, financials, and trading metrics.

```json
{
  "name": "get-ticker-info",
  "arguments": {
    "symbol": "AAPL"
  }
}
```

#### 2. **get-ticker-news**

Get recent news articles for a stock symbol.

```json
{
  "name": "get-ticker-news",
  "arguments": {
    "symbol": "AAPL",
    "count": 10
  }
}
```

#### 3. **search**

Search for stocks, ETFs, and other financial instruments with related news.

```json
{
  "name": "search",
  "arguments": {
    "query": "Apple Inc",
    "count": 10
  }
}
```

#### 4. **get-top-entities**

Get top performing entities in a sector.

```json
{
  "name": "get-top-entities",
  "arguments": {
    "entity_type": "companies", // Options: "etfs", "mutual_funds", "companies", "growth_companies", "performing_companies"
    "sector": "technology", // See supported sectors below
    "count": 10
  }
}
```

Supported sectors:

- basic-materials
- communication-services
- consumer-cyclical
- consumer-defensive
- energy
- financial-services
- healthcare
- industrials
- real-estate
- technology
- utilities

#### 5. **get-price-history**

Get historical price data with customizable periods and intervals.

```json
{
  "name": "get-price-history",
  "arguments": {
    "symbol": "AAPL",
    "period": "1y", // Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
    "interval": "1d" // Options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
  }
}
```

#### 6. **ticker-option-chain**

Get option chain data for stocks.

```json
{
  "name": "ticker-option-chain",
  "arguments": {
    "symbol": "AAPL",
    "option_type": "call", // Options: "call", "put", "both"
    "date": "2024-01-19" // Optional: YYYY-MM-DD format
  }
}
```

#### 7. **ticker-earning**

Get earnings data including historical and upcoming earnings.

```json
{
  "name": "ticker-earning",
  "arguments": {
    "symbol": "AAPL",
    "period": "annual", // Options: "annual", "quarterly"
    "date": "2023-12-31" // Optional: YYYY-MM-DD format
  }
}
```

## 🧪 Testing

### Using MCP Inspector

```bash
npx @modelcontextprotocol/inspector yahoo-finance-server
```

### Manual Testing

```bash
python -c "
import asyncio
from yahoo_finance_server.helper import get_ticker_info

async def test():
    info = await get_ticker_info('AAPL')
    print(f'✅ Stock: {info[\"longName\"]}')

asyncio.run(test())
"
```

## 📋 Requirements

- Python 3.11+
- yfinance==0.2.62
- requests>=2.31.0
- pandas>=2.0.0
- mcp>=1.9.3

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🔗 Links

- **Homepage**: [GitHub Repository](https://github.com/AgentX-ai/AgentX-mcp-servers/tree/main/yahoo_finance_server)
- **Issues**: [Report Issues](https://github.com/AgentX-ai/AgentX-mcp-servers/issues)
- **MCP Documentation**: [Model Context Protocol](https://modelcontextprotocol.io)

---

**Made with ❤️ for the finance community**
