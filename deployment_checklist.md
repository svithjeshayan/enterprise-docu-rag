# ============================================
# deployment_checklist.md
# ============================================
# Deployment Checklist

## Pre-Deployment

- [ ] Set OPENAI_API_KEY in environment or .streamlit/secrets.toml
- [ ] Install all dependencies: `pip install -r requirements.txt`
- [ ] Install Tesseract OCR and set correct path in config
- [ ] Install Poppler and set correct path in config
- [ ] Create documents folder and add initial PDFs
- [ ] Run `python config.py` to validate configuration
- [ ] Test locally: `streamlit run app_improved.py`

## Production Deployment

- [ ] Use PostgreSQL instead of SQLite for multi-user support
- [ ] Set up proper authentication (OAuth, LDAP, etc.)
- [ ] Configure reverse proxy (Nginx) with HTTPS
- [ ] Set up monitoring (Prometheus, Grafana)
- [ ] Configure log rotation
- [ ] Set up automated backups for database and vector store
- [ ] Implement proper secrets management (AWS Secrets Manager, etc.)
- [ ] Set resource limits (memory, CPU)
- [ ] Configure auto-scaling if needed
- [ ] Set up error alerting (email, Slack, PagerDuty)

## Security

- [ ] Enable HTTPS only
- [ ] Implement rate limiting at load balancer level
- [ ] Set up WAF (Web Application Firewall)
- [ ] Regular security audits
- [ ] Keep dependencies updated
- [ ] Implement audit logging
- [ ] Set up IP whitelisting if needed
- [ ] Configure CORS properly

## Monitoring

- [ ] Track API usage and costs
- [ ] Monitor response times
- [ ] Track error rates
- [ ] Monitor vector store size
- [ ] Set up alerts for failures
- [ ] Track user engagement metrics