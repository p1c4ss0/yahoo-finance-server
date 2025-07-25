import asyncio
import json

from mcp.server.models import InitializationOptions
import mcp.types as types
from mcp.server import NotificationOptions, Server
from pydantic import AnyUrl
import mcp.server.stdio

# Import helper functions for Yahoo Finance functionality
from .helper import (
    get_ticker_info,
    get_ticker_news,
    search_yahoo_finance,
    get_top_entities,
    get_price_history,
    get_ticker_option_chain,
    get_ticker_earnings,
    # Trading bot specific functions
    get_indian_market_status,
    get_nse_sector_performance,
    get_enhanced_ticker_info_for_trading,
    get_trading_news_filtered,
)

# Initialize the MCP server
server = Server("yahoo_finance_server")


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """
    List available resources.
    Currently no resources are exposed by this server.
    """
    return []


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> str:
    """
    Read a specific resource by its URI.
    Currently no resources are supported.
    """
    raise ValueError(f"Unsupported resource URI: {uri}")


@server.list_prompts()
async def handle_list_prompts() -> list[types.Prompt]:
    """
    List available prompts.
    Currently no prompts are exposed by this server.
    """
    return []


@server.get_prompt()
async def handle_get_prompt(
    name: str, arguments: dict[str, str] | None
) -> types.GetPromptResult:
    """
    Generate a prompt by name.
    Currently no prompts are supported.
    """
    raise ValueError(f"Unknown prompt: {name}")


@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """
    List available Yahoo Finance tools.
    """
    return [
        types.Tool(
            name="get-ticker-info",
            description="Retrieve comprehensive stock data including company info, financials, trading metrics and governance data",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')",
                    }
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="get-ticker-news",
            description="Fetch recent news articles related to a specific stock symbol with title, content, and source details",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol to get news for",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of news articles to fetch (default: 10, maximum: 50)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="search",
            description="Search Yahoo Finance for stocks, ETFs, and other financial instruments",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (company name, ticker symbol, etc.)",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of search results to return (default: 10, maximum: 25)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 25,
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="get-top-entities",
            description="Get top entities (ETFs, mutual funds, companies, growth companies, or performing companies) in a sector",
            inputSchema={
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "enum": [
                            "etfs",
                            "mutual_funds",
                            "companies",
                            "growth_companies",
                            "performing_companies",
                        ],
                        "description": "Type of entities to retrieve",
                    },
                    "sector": {
                        "type": "string",
                        "description": "Sector name (technology, healthcare, financial, energy, consumer, industrial)",
                        "default": "",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of entities to return (default: 10, maximum: 20)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 20,
                    },
                },
                "required": ["entity_type"],
            },
        ),
        types.Tool(
            name="get-price-history",
            description="Fetch historical price data for a given stock symbol over a specified period and interval",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "period": {
                        "type": "string",
                        "enum": [
                            "1d",
                            "5d",
                            "1mo",
                            "3mo",
                            "6mo",
                            "1y",
                            "2y",
                            "5y",
                            "10y",
                            "ytd",
                            "max",
                        ],
                        "description": "Period to fetch data for",
                        "default": "1y",
                    },
                    "interval": {
                        "type": "string",
                        "enum": [
                            "1m",
                            "2m",
                            "5m",
                            "15m",
                            "30m",
                            "60m",
                            "90m",
                            "1h",
                            "1d",
                            "5d",
                            "1wk",
                            "1mo",
                            "3mo",
                        ],
                        "description": "Data interval",
                        "default": "1d",
                    },
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="ticker-option-chain",
            description="Get most recent or around certain date option chain data. Parameters include call or put, and date. If no date, use most recent top 10 day forward dates",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "option_type": {
                        "type": "string",
                        "enum": ["call", "put", "both"],
                        "description": "Type of options to retrieve",
                        "default": "both",
                    },
                    "date": {
                        "type": "string",
                        "description": "Specific expiration date in YYYY-MM-DD format. If not provided, uses most recent available dates",
                        "default": None,
                    },
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="ticker-earning",
            description="Get earnings data including annual or quarterly data, and upcoming earnings dates. Parameters include annual or quarter, and date. If no date, use most recent, also include the date of upcoming earning time if available",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["annual", "quarterly"],
                        "description": "Earnings period to retrieve",
                        "default": "annual",
                    },
                    "date": {
                        "type": "string",
                        "description": "Specific date in YYYY-MM-DD format. If not provided, uses most recent data",
                        "default": None,
                    },
                },
                "required": ["symbol"],
            },
        ),
        # Trading Bot Specific Tools
        types.Tool(
            name="get-indian-market-status",
            description="Get Indian market (NSE) status including current time, trading hours, and market open/close status",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.Tool(
            name="get-nse-sector-performance",
            description="Get NSE sector performance data using sector ETFs with momentum scores for trading decisions",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.Tool(
            name="get-enhanced-ticker-info-for-trading",
            description="Get enhanced ticker information with trading-specific metrics like SMA comparison, volume surges, and volatility indicators",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol (automatically adds .NS for Indian stocks)",
                    }
                },
                "required": ["symbol"],
            },
        ),
        types.Tool(
            name="get-trading-news-filtered",
            description="Get news filtered for trading relevance with relevance scores, focusing on market-moving news",
            inputSchema={
                "type": "object",
                "properties": {
                    "symbol": {
                        "type": "string",
                        "description": "Stock ticker symbol",
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of filtered news articles to return (default: 10)",
                        "default": 10,
                        "minimum": 1,
                        "maximum": 50,
                    },
                },
                "required": ["symbol"],
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    Handle Yahoo Finance tool execution requests.
    """
    if name == "get-ticker-info":
        return await _handle_get_ticker_info(arguments)
    elif name == "get-ticker-news":
        return await _handle_get_ticker_news(arguments)
    elif name == "search":
        return await _handle_search(arguments)
    elif name == "get-top-entities":
        return await _handle_get_top_entities(arguments)
    elif name == "get-price-history":
        return await _handle_get_price_history(arguments)
    elif name == "ticker-option-chain":
        return await _handle_ticker_option_chain(arguments)
    elif name == "ticker-earning":
        return await _handle_ticker_earning(arguments)
    # Trading bot specific tools
    elif name == "get-indian-market-status":
        return await _handle_get_indian_market_status(arguments)
    elif name == "get-nse-sector-performance":
        return await _handle_get_nse_sector_performance(arguments)
    elif name == "get-enhanced-ticker-info-for-trading":
        return await _handle_get_enhanced_ticker_info_for_trading(arguments)
    elif name == "get-trading-news-filtered":
        return await _handle_get_trading_news_filtered(arguments)
    else:
        raise ValueError(f"Unknown tool: {name}")


async def _handle_get_ticker_info(arguments: dict | None) -> list[types.TextContent]:
    """
    Handle get-ticker-info tool execution using only fast_info fields.
    """
    if not arguments or not arguments.get("symbol"):
        raise ValueError("Symbol is required for ticker info retrieval")

    try:
        symbol = arguments["symbol"].upper()
        ticker_info = await get_ticker_info(symbol)

        # ticker_info is already a JSON string from helper.py
        return [
            types.TextContent(
                type="text",
                text=ticker_info,
            )
        ]

    except Exception as e:
        error_response = json.dumps(
            {"symbol": arguments.get("symbol", "unknown"), "error": str(e)}
        )
        return [
            types.TextContent(
                type="text",
                text=error_response,
            )
        ]


async def _handle_get_ticker_news(arguments: dict | None) -> list[types.TextContent]:
    """
    Handle get-ticker-news tool execution.
    """
    if not arguments or not arguments.get("symbol"):
        raise ValueError("Symbol is required for news retrieval")

    try:
        symbol = arguments["symbol"].upper()
        count = arguments.get("count", 10)

        news_data = await get_ticker_news(symbol, count)

        if not news_data.get("news"):
            return [
                types.TextContent(
                    type="text",
                    text=f"📰 No news found for {symbol}",
                )
            ]

        # Format the response nicely
        news_text = f"""📰 **Recent News for {symbol}** ({news_data['news_count']} articles)

"""

        for i, article in enumerate(news_data["news"], 1):
            summary_text = (
                f"\n• **Summary:** {article['summary']}"
                if article.get("summary") and article["summary"].strip()
                else ""
            )
            news_text += f"""**{i}. {article['title']}**
• **Publisher:** {article['publisher']}
• **Published:** {article['published']}
• **Link:** {article['link']}{summary_text}

"""

        return [
            types.TextContent(
                type="text",
                text=news_text,
            )
        ]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error retrieving news for {arguments.get('symbol', 'unknown')}: {str(e)}",
            )
        ]


async def _handle_search(arguments: dict | None) -> list[types.TextContent]:
    """
    Handle search tool execution.
    """
    if not arguments or not arguments.get("query"):
        raise ValueError("Query is required for search")

    try:
        query = arguments["query"]
        count = arguments.get("count", 10)

        search_data = await search_yahoo_finance(query, count)

        if not search_data.get("results") and not search_data.get("news"):
            return [
                types.TextContent(
                    type="text",
                    text=f"🔍 No results found for '{query}'",
                )
            ]

        return [
            types.TextContent(
                type="text",
                text=json.dumps(search_data),
            )
        ]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error searching for '{arguments.get('query', 'unknown')}': {str(e)}",
            )
        ]


async def _handle_get_top_entities(arguments: dict | None) -> list[types.TextContent]:
    """
    Handle get-top-entities tool execution.
    """
    if not arguments or not arguments.get("entity_type"):
        raise ValueError("Entity type is required")

    try:
        entity_type = arguments["entity_type"]
        sector = arguments.get("sector", "")
        count = arguments.get("count", 10)

        entities_data = await get_top_entities(entity_type, sector, count)

        if not entities_data.get("results"):
            return [
                types.TextContent(
                    type="text",
                    text=f"🏆 No {entity_type} found for sector '{sector}'",
                )
            ]

        return [
            types.TextContent(
                type="text",
                text=json.dumps(entities_data),
            )
        ]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error retrieving top entities: {str(e)}",
            )
        ]


async def _handle_get_price_history(arguments: dict | None) -> list[types.TextContent]:
    """
    Handle get-price-history tool execution.
    """
    if not arguments or not arguments.get("symbol"):
        raise ValueError("Symbol is required for price history")

    try:
        symbol = arguments["symbol"].upper()
        period = arguments.get("period", "1y")
        interval = arguments.get("interval", "1d")

        history_data = await get_price_history(symbol, period, interval)

        if not history_data.get("data"):
            return [
                types.TextContent(
                    type="text",
                    text=f"📈 No price history found for {symbol}",
                )
            ]

        # Format the response nicely - show last 10 data points
        history_text = f"""📈 **Price History for {symbol}**
**Period:** {period} | **Interval:** {interval} | **Data Points:** {history_data['count']}

**Recent Data (Last 10 points):**
"""

        recent_data = history_data["data"][-10:]  # Get last 10 data points
        for data_point in recent_data:
            history_text += f"""• **{data_point['date'][:10]}**: Open: ${data_point['open']:.2f}, High: ${data_point['high']:.2f}, Low: ${data_point['low']:.2f}, Close: ${data_point['close']:.2f}, Volume: {data_point['volume']:,}
"""

        # Add summary statistics
        closes = [d["close"] for d in history_data["data"] if d["close"] is not None]
        if closes:
            min_price = min(closes)
            max_price = max(closes)
            avg_price = sum(closes) / len(closes)

            history_text += f"""
**Summary Statistics:**
• **Period Low:** ${min_price:.2f}
• **Period High:** ${max_price:.2f}
• **Average Price:** ${avg_price:.2f}
• **Total Change:** {((closes[-1] - closes[0]) / closes[0] * 100):.2f}%
"""

        return [
            types.TextContent(
                type="text",
                text=history_text,
            )
        ]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error retrieving price history for {arguments.get('symbol', 'unknown')}: {str(e)}",
            )
        ]


async def _handle_ticker_option_chain(
    arguments: dict | None,
) -> list[types.TextContent]:
    """
    Handle ticker-option-chain tool execution.
    """
    if not arguments or not arguments.get("symbol"):
        raise ValueError("Symbol is required for option chain")

    try:
        symbol = arguments["symbol"].upper()
        option_type = arguments.get("option_type", "both")
        date = arguments.get("date")

        options_data = await get_ticker_option_chain(symbol, option_type, date)

        if options_data.get("error"):
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ {options_data['error']}",
                )
            ]

        # Format the response nicely
        options_text = f"""⚡ **Option Chain for {symbol}**
**Expiration Date:** {options_data.get('expiration_date', 'N/A')}
**Available Expirations:** {', '.join(options_data.get('available_expirations', [])[:5])}

"""

        if option_type in ["call", "both"] and "calls" in options_data:
            options_text += "**📈 CALL Options:**\n"
            calls = options_data["calls"][:10]  # Show first 10
            for call in calls:
                options_text += f"""• Strike: ${call['strike']:.2f} | Last: ${call['last_price']:.2f} | Bid: ${call['bid']:.2f} | Ask: ${call['ask']:.2f} | Vol: {call['volume']} | OI: {call['open_interest']} | IV: {call['implied_volatility']:.2%}
"""
            options_text += "\n"

        if option_type in ["put", "both"] and "puts" in options_data:
            options_text += "**📉 PUT Options:**\n"
            puts = options_data["puts"][:10]  # Show first 10
            for put in puts:
                options_text += f"""• Strike: ${put['strike']:.2f} | Last: ${put['last_price']:.2f} | Bid: ${put['bid']:.2f} | Ask: ${put['ask']:.2f} | Vol: {put['volume']} | OI: {put['open_interest']} | IV: {put['implied_volatility']:.2%}
"""

        return [
            types.TextContent(
                type="text",
                text=options_text,
            )
        ]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error retrieving option chain for {arguments.get('symbol', 'unknown')}: {str(e)}",
            )
        ]


async def _handle_ticker_earning(arguments: dict | None) -> list[types.TextContent]:
    """
    Handle ticker-earning tool execution.
    """
    if not arguments or not arguments.get("symbol"):
        raise ValueError("Symbol is required for earnings data")

    try:
        symbol = arguments["symbol"].upper()
        period = arguments.get("period", "annual")
        date = arguments.get("date")

        earnings_data = await get_ticker_earnings(symbol, period, date)

        if earnings_data.get("error"):
            return [
                types.TextContent(
                    type="text",
                    text=f"❌ Error retrieving earnings: {earnings_data['error']}",
                )
            ]

        # Format the response nicely
        earnings_text = f"""💰 **Earnings Data for {symbol}**
**Period:** {period.title()}

**Key Metrics:**
• **Trailing EPS:** ${earnings_data.get('trailing_eps', 0):.2f}
• **Forward EPS:** ${earnings_data.get('forward_eps', 0):.2f}
• **P/E Ratio:** {earnings_data.get('pe_ratio', 0):.2f}
• **Forward P/E:** {earnings_data.get('forward_pe', 0):.2f}

"""

        # Historical earnings data
        if earnings_data.get("earnings_data"):
            earnings_text += f"**Historical {period.title()} Earnings:**\n"
            recent_earnings = earnings_data["earnings_data"][:4]  # Show last 4
            for earning in recent_earnings:
                earnings_text += f"""• **{earning['date']}**: Revenue: ${earning['total_revenue']:,} | Net Income: ${earning['net_income']:,} | EBITDA: ${earning['ebitda']:,}
"""
            earnings_text += "\n"

        # Upcoming earnings
        if earnings_data.get("upcoming_earnings"):
            earnings_text += "**Upcoming Earnings:**\n"
            for upcoming in earnings_data["upcoming_earnings"]:
                earnings_text += f"""• **Date:** {upcoming['earnings_date']} | **EPS Est:** ${upcoming['eps_estimate']:.2f}
"""

        return [
            types.TextContent(
                type="text",
                text=earnings_text,
            )
        ]

    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error retrieving earnings for {arguments.get('symbol', 'unknown')}: {str(e)}",
            )
        ]


# Trading Bot Specific Handler Functions

async def _handle_get_indian_market_status(arguments: dict | None) -> list[types.TextContent]:
    """
    Handle get-indian-market-status tool execution.
    """
    try:
        market_status = await get_indian_market_status()
        
        status_text = f"""📊 **Indian Market (NSE) Status**

🕒 **Current Time (IST):** {market_status['current_time_ist']}
📈 **Market Status:** {'🟢 OPEN' if market_status['is_market_open'] else '🔴 CLOSED'}
⏰ **Trading Hours:** {market_status['trading_hours']['start']} - {market_status['trading_hours']['end']} IST

"""
        if market_status.get('next_open'):
            status_text += f"📅 **Next Opening:** {market_status['next_open']}\n"
        
        return [
            types.TextContent(
                type="text",
                text=status_text,
            )
        ]
    
    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error retrieving Indian market status: {str(e)}",
            )
        ]


async def _handle_get_nse_sector_performance(arguments: dict | None) -> list[types.TextContent]:
    """
    Handle get-nse-sector-performance tool execution.
    """
    try:
        sector_data = await get_nse_sector_performance()
        
        performance_text = f"""📊 **NSE Sector Performance Analysis**

🏆 **Top Performing Sector:** {sector_data.get('top_performing_sector', 'N/A')}
📅 **Analysis Time:** {sector_data['analysis_time']}

**Sector Rankings by Momentum Score:**

"""
        
        for i, (sector, data) in enumerate(sector_data.get('sectors', {}).items(), 1):
            emoji = "🟢" if data['daily_change_percent'] > 0 else "🔴"
            performance_text += f"""**{i}. {sector}** {emoji}
   • Daily Change: {data['daily_change_percent']}%
   • Volume Ratio: {data['volume_ratio']}x
   • Momentum Score: {data['momentum_score']}
   • ETF: {data['etf_symbol']} (₹{data['latest_price']})

"""
        
        return [
            types.TextContent(
                type="text",
                text=performance_text,
            )
        ]
    
    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error retrieving NSE sector performance: {str(e)}",
            )
        ]


async def _handle_get_enhanced_ticker_info_for_trading(arguments: dict | None) -> list[types.TextContent]:
    """
    Handle get-enhanced-ticker-info-for-trading tool execution.
    """
    if not arguments or not arguments.get("symbol"):
        raise ValueError("Symbol is required for enhanced ticker info")
    
    try:
        symbol = arguments["symbol"].upper()
        enhanced_info = await get_enhanced_ticker_info_for_trading(symbol)
        
        # Extract key information
        company_name = enhanced_info.get('longName', enhanced_info.get('shortName', symbol))
        current_price = enhanced_info.get('currentPrice', enhanced_info.get('regularMarketPrice', 'N/A'))
        trading_metrics = enhanced_info.get('trading_metrics', {})
        
        info_text = f"""📈 **Enhanced Trading Info: {company_name} ({enhanced_info.get('symbol_normalized', symbol)})**

💰 **Current Price:** ₹{current_price}
📊 **Market Cap:** ₹{enhanced_info.get('marketCap', 'N/A'):,} 

**🎯 Trading Metrics:**
• **Price vs SMA20:** {trading_metrics.get('price_vs_sma20_percent', 'N/A')}% {'🟢' if trading_metrics.get('is_above_sma20') else '🔴'}
• **Volume Surge:** {trading_metrics.get('volume_surge_ratio', 'N/A')}x {'⚡' if trading_metrics.get('volume_anomaly') else ''}
• **ATR (Volatility):** {trading_metrics.get('atr_percent', 'N/A')}% {'🌋' if trading_metrics.get('high_volatility') else ''}

**📈 Key Metrics:**
• **P/E Ratio:** {enhanced_info.get('trailingPE', enhanced_info.get('forwardPE', 'N/A'))}
• **52W High:** ₹{enhanced_info.get('fiftyTwoWeekHigh', 'N/A')}
• **52W Low:** ₹{enhanced_info.get('fiftyTwoWeekLow', 'N/A')}
• **Average Volume:** {enhanced_info.get('averageVolume', 'N/A'):,}

**📊 Financial Health:**
• **Revenue:** ₹{enhanced_info.get('totalRevenue', 'N/A'):,}
• **Debt/Equity:** {enhanced_info.get('debtToEquity', 'N/A')}
• **ROE:** {enhanced_info.get('returnOnEquity', 'N/A')}%

⏰ **Last Updated:** {trading_metrics.get('last_updated', 'N/A')}
"""
        
        return [
            types.TextContent(
                type="text",
                text=info_text,
            )
        ]
    
    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error retrieving enhanced ticker info for {arguments.get('symbol', 'unknown')}: {str(e)}",
            )
        ]


async def _handle_get_trading_news_filtered(arguments: dict | None) -> list[types.TextContent]:
    """
    Handle get-trading-news-filtered tool execution.
    """
    if not arguments or not arguments.get("symbol"):
        raise ValueError("Symbol is required for filtered news")
    
    try:
        symbol = arguments["symbol"].upper()
        count = arguments.get("count", 10)
        
        news_data = await get_trading_news_filtered(symbol, count)
        
        if not news_data.get('news'):
            return [
                types.TextContent(
                    type="text",
                    text=f"📰 No trading-relevant news found for {symbol}",
                )
            ]
        
        news_text = f"""📰 **Trading-Relevant News for {symbol}**

🔍 **Filter Applied:** {news_data.get('filter_applied', 'N/A')}
📊 **Results:** {news_data.get('filtered_count', 0)} relevant articles from {news_data.get('original_count', 0)} total

"""
        
        for i, article in enumerate(news_data['news'], 1):
            relevance_score = article.get('trading_relevance_score', 0)
            relevance_emoji = "🔥" if relevance_score >= 5 else "📈" if relevance_score >= 3 else "📊"
            
            news_text += f"""**{i}. {article.get('title', 'No title')}** {relevance_emoji}
   🏢 **Source:** {article.get('publisher', 'Unknown')}
   📅 **Published:** {article.get('published', 'Unknown')}
   🎯 **Relevance Score:** {relevance_score}/10
   🔗 **Link:** {article.get('link', 'N/A')}
"""
            if article.get('summary'):
                news_text += f"   📄 **Summary:** {article['summary'][:200]}...\n"
            news_text += "\n"
        
        return [
            types.TextContent(
                type="text",
                text=news_text,
            )
        ]
    
    except Exception as e:
        return [
            types.TextContent(
                type="text",
                text=f"❌ Error retrieving filtered news for {arguments.get('symbol', 'unknown')}: {str(e)}",
            )
        ]


async def main():
    """Main entry point for the Yahoo Finance MCP server."""
    # Use stdio transport for MCP communication
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="trading-bot-yahoo-finance-server",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
