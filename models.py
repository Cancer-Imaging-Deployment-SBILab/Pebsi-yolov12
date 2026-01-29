import enum
from datetime import datetime, timezone
import pytz

from sqlalchemy import (
    Boolean,
    Column,
    Integer,
    String,
    DateTime,
    ForeignKey,
    Enum,
    Date,
    Float,
    text,
    func,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID, JSON
from sqlalchemy.orm import relationship

from database import Base


# ======================================================
# Utility
# ======================================================
def convert_utc_to_timezone(utc_dt: datetime, tz_str: str = "Asia/Kolkata") -> datetime:
    if utc_dt is None:
        return None
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    return utc_dt.astimezone(pytz.timezone(tz_str))


# ======================================================
# Enums
# ======================================================
class RoleEnum(enum.Enum):
    Admin = "Admin"
    Editor = "Editor"
    Viewer = "Viewer"
    SuperAdmin = "SuperAdmin"


class StatusEnum(enum.Enum):
    active = "active"
    inactive = "inactive"


class GenderEnum(enum.Enum):
    male = "male"
    female = "female"
    other = "other"
    ratherNotSay = "ratherNotSay"


class TestEnum(enum.Enum):
    cbc = "cbc"


# ======================================================
# User
# ======================================================
class User(Base):
    __tablename__ = "users"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    name = Column(String, nullable=False, index=True)
    phone_no = Column(String, nullable=False)
    employee_id = Column(String, unique=True, nullable=False, index=True)
    password = Column(String, nullable=False)
    role = Column(Enum(RoleEnum), nullable=False, index=True)
    status = Column(Enum(StatusEnum), nullable=False, server_default=text("'active'"))
    failed_login_attempts = Column(Integer, nullable=False, server_default=text("0"))

    reports = relationship("Report", back_populates="generated_by")
    assignments_created = relationship(
        "TestAssignment",
        back_populates="assigned_by",
        foreign_keys="TestAssignment.assigned_by_user_id",
    )
    assignments_received = relationship(
        "TestAssignment",
        back_populates="assigned_to",
        foreign_keys="TestAssignment.assigned_to_user_id",
    )
    audit_logs = relationship("AuditLog", back_populates="user")
    sessions = relationship("Session", back_populates="user")


# ======================================================
# Patient
# ======================================================
class Patient(Base):
    __tablename__ = "patients"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    name = Column(String, nullable=False, index=True)
    uhid = Column(String, unique=True, nullable=False, index=True)
    dob = Column(Date, nullable=False, index=True)
    phone_no = Column(String, nullable=False)
    aadhar = Column(String, nullable=False, index=True)
    gender = Column(Enum(GenderEnum), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    tests = relationship("Test", back_populates="patient")
    samples = relationship("Sample", back_populates="patient")
    reports = relationship("Report", back_populates="patient")


# ======================================================
# Test
# ======================================================
class Test(Base):
    __tablename__ = "tests"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    patient_id = Column(
        UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False, index=True
    )
    custom_sample_id = Column(String, unique=True, nullable=False, index=True)
    condition = Column(String, nullable=False)
    in_use = Column(Boolean, nullable=False, default=False)
    test_name = Column(Enum(TestEnum), nullable=False, index=True)
    test_datetime = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="tests")
    samples = relationship("Sample", back_populates="test")
    report = relationship("Report", back_populates="test", uselist=False)
    blood_counts = relationship("CompleteBloodCount", back_populates="test")
    assignments = relationship("TestAssignment", back_populates="test")


# ======================================================
# Sample
# ======================================================
class Sample(Base):
    __tablename__ = "samples"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    patient_id = Column(
        UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False, index=True
    )
    test_id = Column(
        UUID(as_uuid=True), ForeignKey("tests.id"), nullable=False, index=True
    )
    isProcessed = Column(Boolean, nullable=False, default=False)
    scale = Column(String, nullable=False)

    sample_location = Column(String, nullable=False)
    sample_datetime = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="samples")
    test = relationship("Test", back_populates="samples")
    cbc = relationship("CompleteBloodCount", back_populates="sample", uselist=False)
    annotations = relationship("Annotation", back_populates="sample")

    __table_args__ = (Index("idx_sample_test_processed", "test_id", "isProcessed"),)


# ======================================================
# Report
# ======================================================
class Report(Base):
    __tablename__ = "reports"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    patient_id = Column(
        UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False, index=True
    )
    test_id = Column(
        UUID(as_uuid=True), ForeignKey("tests.id"), nullable=False, index=True
    )
    generated_by_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )

    diagnosis = Column(String, nullable=False)
    report_location = Column(String, nullable=False)
    report_datetime = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="reports")
    test = relationship("Test", back_populates="report")
    generated_by = relationship("User", back_populates="reports")


# ======================================================
# Complete Blood Count
# ======================================================
class CompleteBloodCount(Base):
    __tablename__ = "complete_blood_counts"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    sample_id = Column(
        UUID(as_uuid=True),
        ForeignKey("samples.id"),
        unique=True,
        nullable=False,
        index=True,
    )
    test_id = Column(
        UUID(as_uuid=True), ForeignKey("tests.id"), nullable=False, index=True
    )

    haemoglobin = Column(Float)
    pcv = Column(Float)
    rbc_count = Column(Float)
    mcv = Column(Float)
    mch = Column(Float)
    mchc = Column(Float)
    rdw_cv = Column(Float)
    rdw_sd = Column(Float)
    platelet_count = Column(Float)
    tlc = Column(Float)
    neutrophil = Column(Float)
    lymphocyte = Column(Float)
    eosinophil = Column(Float)
    monocyte = Column(Float)
    basophil = Column(Float)
    abs_neutrophil_count = Column(Float)
    abs_monocyte_count = Column(Float)
    abs_lymphocyte_count = Column(Float)
    abs_eosinophil_count = Column(Float)
    abs_basophil_count = Column(Float)
    mpv = Column(Float)
    pdw = Column(Float)
    neutrophil_lymphocyte = Column(Float)
    lymphocyte_monocyte = Column(Float)
    pct = Column(Float)
    p_lcc = Column(Float)
    p_lcr = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    sample = relationship("Sample", back_populates="cbc")
    test = relationship("Test", back_populates="blood_counts")


# ======================================================
# Annotation
# ======================================================
class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(String, primary_key=True)
    sample_id = Column(
        UUID(as_uuid=True), ForeignKey("samples.id"), nullable=False, index=True
    )

    source = Column(String, nullable=False)
    type = Column(String, nullable=False)
    tool = Column(String, nullable=False)
    height = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)

    isProcessed = Column(Boolean, nullable=False, default=False)
    isClassDetected = Column(Boolean, nullable=False, default=False)
    isSubClassDetected = Column(Boolean, nullable=False, default=False)
    isSegmented = Column(Boolean, nullable=False, default=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    sample = relationship("Sample", back_populates="annotations")
    annotation_boxes = relationship("AnnotationBox", back_populates="annotation")

    __table_args__ = (
        Index("idx_annotation_workflow_class", "isProcessed", "isClassDetected"),
        Index(
            "idx_annotation_workflow_subclass", "isClassDetected", "isSubClassDetected"
        ),
        Index("idx_annotation_workflow_segment", "isSubClassDetected", "isSegmented"),
        Index("idx_annotation_created_processed", "created_at", "isProcessed"),
        Index("idx_annotation_sample_processed", "sample_id", "isProcessed"),
    )


# ======================================================
# AnnotationBox
# ======================================================
class AnnotationBox(Base):
    __tablename__ = "annotation_boxes"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    annotation_id = Column(
        String, ForeignKey("annotations.id"), nullable=False, index=True
    )
    boxes = Column(JSON, nullable=False)

    class_name = Column(String)
    class_confidence = Column(Float)
    sub_class = Column(String, index=True)
    sub_class_confidence = Column(Float)

    crop_path = Column(String)
    segmentation_polygon = Column(JSON)
    isModelClass = Column(Boolean, nullable=False, default=False)
    isModelSubClass = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    annotation = relationship("Annotation", back_populates="annotation_boxes")
    wbc_sub_class_confidences = relationship(
        "WBCSubClassConfidences", back_populates="annotation_box", uselist=False
    )

    __table_args__ = (
        Index("idx_box_annotation_class", "annotation_id", "class_name"),
        Index("idx_box_class_subclass", "class_name", "sub_class"),
        Index("idx_box_class_confidence", "class_name", "class_confidence"),
        Index("idx_box_created_class", "created_at", "class_name"),
        Index("idx_box_subclass_confidence", "sub_class", "sub_class_confidence"),
        Index("idx_box_annotation_created", "annotation_id", "created_at"),
    )


# ======================================================
# WBC SubClass Confidences
# ======================================================
class WBCSubClassConfidences(Base):
    __tablename__ = "wbc_sub_class_confidences"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    annotation_box_id = Column(
        UUID(as_uuid=True),
        ForeignKey("annotation_boxes.id"),
        unique=True,
        nullable=False,
        index=True,
    )

    neutrophil_confidence = Column(Float)
    lymphocyte_confidence = Column(Float)
    eosinophil_confidence = Column(Float)
    monocyte_confidence = Column(Float)
    basophil_confidence = Column(Float)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    annotation_box = relationship(
        "AnnotationBox", back_populates="wbc_sub_class_confidences"
    )


# ======================================================
# Cell Classes
# ======================================================
class CellClasses(Base):
    __tablename__ = "cell_classes"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    name = Column(String, nullable=False, unique=True, index=True)
    colors = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    sub_classes = relationship("CellSubClasses", back_populates="cell_class")


class CellSubClasses(Base):
    __tablename__ = "cell_sub_classes"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    class_id = Column(
        UUID(as_uuid=True), ForeignKey("cell_classes.id"), nullable=False, index=True
    )
    sub_class_name = Column(String, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    cell_class = relationship("CellClasses", back_populates="sub_classes")


# ======================================================
# Test Assignment
# ======================================================
# This table supports assigning a test to MULTIPLE users.
# Each row represents one user assigned to a test.
# The unique constraint on (test_id, assigned_to_user_id) prevents
# the same user from being assigned to the same test twice.
class TestAssignment(Base):
    __tablename__ = "test_assignments"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    assigned_by_user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    assigned_to_user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    test_id = Column(
        UUID(as_uuid=True), ForeignKey("tests.id"), nullable=False, index=True
    )
    assigned_at = Column(DateTime(timezone=True), server_default=func.now())
    isArchived = Column(Boolean, nullable=False, default=False)
    archived_date = Column(DateTime(timezone=True), nullable=True)

    assigned_by = relationship(
        "User", back_populates="assignments_created", foreign_keys=[assigned_by_user_id]
    )
    assigned_to = relationship(
        "User",
        back_populates="assignments_received",
        foreign_keys=[assigned_to_user_id],
    )
    test = relationship("Test", back_populates="assignments")

    __table_args__ = (
        # Composite index for efficient lookups by assigned user and test
        Index("idx_assignment_to_test", "assigned_to_user_id", "test_id"),
        # Composite index for efficient lookups by test (to find all assigned users)
        Index("idx_assignment_test_users", "test_id", "assigned_to_user_id"),
        # Unique constraint to prevent duplicate assignments (same user assigned to same test twice)
        {"extend_existing": True},
    )

    # Add unique constraint at table level
    __table_args__ = (
        Index("idx_assignment_to_test", "assigned_to_user_id", "test_id"),
        Index("idx_assignment_test_users", "test_id", "assigned_to_user_id"),
        # Unique constraint: a user can only be assigned to a specific test once
        Index(
            "uq_test_assignment_user_test",
            "test_id",
            "assigned_to_user_id",
            unique=True,
        ),
    )


# ======================================================
# Audit Log
# ======================================================
class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    action = Column(String, nullable=False, index=True)
    table_name = Column(String, nullable=False, index=True)
    user_ip_address = Column(
        String, nullable=False, server_default=text("inet_client_addr()")
    )
    old_data = Column(JSON)
    new_data = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="audit_logs")

    __table_args__ = (
        Index("idx_audit_user_created", "user_id", "created_at"),
        Index("idx_audit_table_created", "table_name", "created_at"),
        Index("idx_audit_action_created", "action", "created_at"),
        Index("idx_audit_user_table_action", "user_id", "table_name", "action"),
    )


# ======================================================
# Session
# ======================================================
class Session(Base):
    __tablename__ = "sessions"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    refresh_token_hash = Column(String, nullable=False, index=True)

    user_ip_address = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_activity_at = Column(DateTime(timezone=True), server_default=func.now())

    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked = Column(Boolean, nullable=False, default=False)

    user = relationship("User", back_populates="sessions")
    access_tokens = relationship(
        "AccessToken", back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("idx_session_token_valid", "refresh_token_hash", "revoked", "expires_at"),
        Index("idx_session_user_active", "user_id", "revoked", "expires_at"),
        Index("idx_session_expires_revoked", "expires_at", "revoked"),
    )


# ======================================================
# Access Token
# ======================================================
class AccessToken(Base):
    __tablename__ = "access_tokens"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    token_hash = Column(String, nullable=False, unique=True, index=True)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True
    )
    session_id = Column(
        UUID(as_uuid=True), ForeignKey("sessions.id"), nullable=False, index=True
    )
    role = Column(String, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    revoked = Column(Boolean, nullable=False, default=False)

    session = relationship("Session", back_populates="access_tokens")

    __table_args__ = (
        Index("idx_token_validation", "token_hash", "revoked", "expires_at"),
        Index("idx_token_user_active", "user_id", "revoked", "expires_at"),
        Index("idx_token_expires_revoked", "expires_at", "revoked"),
        Index("idx_token_session_active", "session_id", "revoked"),
    )
