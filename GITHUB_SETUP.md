# GitHub Setup Guide

## 📝 Before You Push

### ✅ Checklist

- [x] `.gitignore` file created (protects sensitive files)
- [x] `.env` file will NOT be committed (contains your secrets)
- [x] README.md is complete and informative
- [x] Unnecessary files removed

### ⚠️ Important: Protect Your Credentials

Your `.env` file contains sensitive information and is already in `.gitignore`. 
**Never commit this file to GitHub!**

## 🚀 Push to GitHub

### Option 1: Create New Repository on GitHub First (Recommended)

1. **Go to GitHub** and create a new repository:
   - Go to https://github.com/new
   - Name: `mongodb-chatbot` (or your preferred name)
   - Description: "Intelligent chatbot using MongoDB Vector Search and OpenAI"
   - Choose Public or Private
   - **DO NOT** initialize with README (we already have one)
   - Click "Create repository"

2. **Initialize Git in your project** (if not already done):
   ```bash
   git init
   ```

3. **Add all files:**
   ```bash
   git add .
   ```

4. **Check what will be committed** (make sure .env is NOT listed):
   ```bash
   git status
   ```
   
   You should see files like:
   - ✅ README.md
   - ✅ requirements.txt
   - ✅ main.py
   - ✅ src/
   - ❌ .env (should NOT appear - it's ignored)
   - ❌ logs/ (should NOT appear - it's ignored)
   - ❌ documents/ (should NOT appear - it's ignored)

5. **Commit your changes:**
   ```bash
   git commit -m "Initial commit: MongoDB Chatbot System"
   ```

6. **Add remote repository:**
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   ```
   
   Replace `YOUR_USERNAME` and `YOUR_REPO_NAME` with your actual GitHub username and repository name.

7. **Push to GitHub:**
   ```bash
   git branch -M main
   git push -u origin main
   ```

### Option 2: Push to Existing Repository

If you already have a repository:

```bash
# Initialize git (if not done)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: MongoDB Chatbot System"

# Add remote (replace with your repo URL)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# Push
git branch -M main
git push -u origin main
```

## 🔐 Security Check

Before pushing, verify these files are NOT being committed:

```bash
# This command should show these files are ignored:
git status --ignored
```

Should see:
- `.env` (ignored)
- `logs/` (ignored)
- `documents/` (ignored)
- `__pycache__/` (ignored)
- `sleepdrop/` (ignored - virtual environment)

## 📦 What Gets Pushed

These files WILL be committed:
- ✅ Source code (`src/`)
- ✅ Scripts (`scripts/`)
- ✅ Configuration templates (`.env.example`)
- ✅ Documentation (`README.md`, `COMMANDS.md`)
- ✅ Dependencies (`requirements.txt`)
- ✅ Main entry point (`main.py`)
- ✅ `.gitignore` file

These files will NOT be committed (protected by .gitignore):
- ❌ `.env` (your credentials)
- ❌ `logs/` (log files)
- ❌ `documents/` (your PDFs)
- ❌ `__pycache__/` (Python cache)
- ❌ `sleepdrop/` (virtual environment)

## 🎯 After Pushing

### Update Repository Settings

1. **Add Topics** (on GitHub repository page):
   - `chatbot`
   - `mongodb`
   - `openai`
   - `vector-search`
   - `fastapi`
   - `python`
   - `langgraph`

2. **Add Description**:
   "Intelligent chatbot using MongoDB Vector Search and OpenAI GPT-4"

3. **Add Website** (optional):
   Your deployed URL if you deploy it

### Create a Good README Badge (Optional)

Add badges to your README.md:

```markdown
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green.svg)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)
```

## 🔄 Future Updates

When you make changes:

```bash
# Check what changed
git status

# Add changes
git add .

# Commit with a descriptive message
git commit -m "Add feature: description of what you added"

# Push to GitHub
git push
```

## 🆘 Troubleshooting

### "Permission denied" error
Use HTTPS URL or set up SSH keys:
```bash
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
```

### Accidentally committed .env file
If you accidentally committed your .env file:

```bash
# Remove from git but keep local file
git rm --cached .env

# Commit the removal
git commit -m "Remove .env from repository"

# Push
git push

# IMPORTANT: Rotate your API keys immediately!
# - Get new OpenAI API key
# - Update MongoDB password
```

### Want to see what's ignored
```bash
git status --ignored
```

## ✅ Verification

After pushing, visit your GitHub repository and verify:
1. ✅ README.md displays correctly
2. ✅ `.env` file is NOT visible
3. ✅ All source code is present
4. ✅ `.gitignore` is working

## 🎉 Done!

Your project is now on GitHub! Share the link with others, and they can:
1. Clone your repository
2. Set up their own `.env` file
3. Run the chatbot with their own credentials

---

**Remember**: Never share your `.env` file or commit it to GitHub!
