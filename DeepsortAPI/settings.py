from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent


DEBUG = True

ALLOWED_HOSTS = ['*']


INSTALLED_APPS = [
    "django.contrib.staticfiles",
    "stream.apps.StreamConfig",
    "rest_framework",
    "corsheaders"
]

CORS_ORIGIN_WHITELIST = ['http://127.0.0.1:3000', 'http://localhost:3000']
CORS_ALLOW_CREDENTIALS = True

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
    "corsheaders.middleware.CorsMiddleware"
]

ROOT_URLCONF = "DeepsortAPI.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / 'templates']
        ,
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "DeepsortAPI.wsgi.application"


# DATABASES = {
#     'default': {
#         'ENGINE': config.get('DATABASE_ENGINE'),
#         'NAME': config.get('DATABASE_NAME'),
#         'USER': config.get('DATABASE_USER'),
#         'PASSWORD': config.get('DATABASE_PASS'),
#         'HOST': config.get('DATABASE_HOST'),
#         'PORT': config.get('DATABASE_PORT')
#     }
# }


AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator", },
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator", },
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator", },
]


LANGUAGE_CODE = "ko-kr"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


STATIC_URL = "static/"


DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"
