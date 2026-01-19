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


def convert_utc_to_timezone(utc_dt: datetime, tz_str: str = "Asia/Kolkata") -> datetime:
    """
    Convert a UTC datetime to the specified timezone.
    
    Args:
        utc_dt: A datetime object in UTC
        tz_str: Timezone string (e.g., 'Asia/Kolkata', 'America/New_York')
        
    Returns:
        datetime: The datetime converted to the specified timezone
    """
    if utc_dt is None:
        return None
    
    # Ensure the datetime is timezone-aware (UTC)
    if utc_dt.tzinfo is None:
        utc_dt = utc_dt.replace(tzinfo=timezone.utc)
    
    # Convert to target timezone
    target_tz = pytz.timezone(tz_str)
    return utc_dt.astimezone(target_tz)


# Enums
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


# Models
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

    __table_args__ = (
        Index('idx_users_id', 'id'),
        Index('idx_users_name', 'name'),
        Index('idx_users_employee_id', 'employee_id'),
        Index('idx_users_role', 'role'),
    )


# TODO Chaniging for ma'am to be changed
class Patient(Base):
    __tablename__ = "patients"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )

    name = Column(String, nullable=False, index=True)
    uhid = Column(String, unique=True, nullable=False, index=True)
    dob = Column(Date, nullable=False, index=True)
    phone_no = Column(String, nullable=False)
    # TODO: Make Aadhar unique
    aadhar = Column(String, nullable=False, index=True)
    gender = Column(Enum(GenderEnum), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    tests = relationship("Test", back_populates="patient")
    samples = relationship("Sample", back_populates="patient")
    reports = relationship("Report", back_populates="patient")

    __table_args__ = (
        Index('idx_patients_id', 'id'),
        Index('idx_patients_name', 'name'),
        Index('idx_patients_uhid', 'uhid'),
        Index('idx_patients_dob', 'dob'),
        Index('idx_patients_aadhar', 'aadhar'),
    )


class Test(Base):
    __tablename__ = "tests"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False, index=True)
    # For now, we are using a custom sample ID to identify tests
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

    __table_args__ = (
        Index('idx_tests_id', 'id'),
        Index('idx_tests_patient_id', 'patient_id'),
        Index('idx_tests_custom_sample_id', 'custom_sample_id'),
        Index('idx_tests_test_name', 'test_name'),
    )


class Sample(Base):
    __tablename__ = "samples"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False, index=True)
    test_id = Column(UUID(as_uuid=True), ForeignKey("tests.id"), nullable=False, index=True)
    isProcessed = Column(Boolean, nullable=False, default=False, index=True)

    sample_location = Column(String, nullable=False)
    sample_datetime = Column(DateTime(timezone=True), server_default=func.now())

    patient = relationship("Patient", back_populates="samples")
    test = relationship("Test", back_populates="samples")
    cbc = relationship("CompleteBloodCount", back_populates="sample", uselist=False)
    annotations = relationship("Annotation", back_populates="sample")

    __table_args__ = (
        Index('idx_samples_id', 'id'),
        Index('idx_samples_patient_id', 'patient_id'),
        Index('idx_samples_test_id', 'test_id'),
        Index('idx_samples_isProcessed', 'isProcessed'),
    )


class Report(Base):
    __tablename__ = "reports"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    patient_id = Column(UUID(as_uuid=True), ForeignKey("patients.id"), nullable=False, index=True)
    test_id = Column(UUID(as_uuid=True), ForeignKey("tests.id"), nullable=False, index=True)

    diagnosis = Column(String, nullable=False)
    report_location = Column(String, nullable=False)
    report_datetime = Column(DateTime(timezone=True), server_default=func.now())
    generated_by_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)

    patient = relationship("Patient", back_populates="reports")
    test = relationship("Test", back_populates="report")
    generated_by = relationship("User", back_populates="reports")

    __table_args__ = (
        Index('idx_reports_id', 'id'),
        Index('idx_reports_patient_id', 'patient_id'),
        Index('idx_reports_test_id', 'test_id'),
        Index('idx_reports_generated_by_id', 'generated_by_id'),
    )


class CompleteBloodCount(Base):
    __tablename__ = "complete_blood_counts"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    sample_id = Column(
        UUID(as_uuid=True), ForeignKey("samples.id"), unique=True, nullable=False, index=True
    )
    test_id = Column(UUID(as_uuid=True), ForeignKey("tests.id"), nullable=False, index=True)

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

    __table_args__ = (
        Index('idx_cbc_id', 'id'),
        Index('idx_cbc_sample_id', 'sample_id'),
        Index('idx_cbc_test_id', 'test_id'),
    )


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(String, primary_key=True)
    sample_id = Column(UUID(as_uuid=True), ForeignKey("samples.id"), nullable=False, index=True)
    source = Column(String, nullable=False)
    type = Column(String, nullable=False)
    tool = Column(String, nullable=False)
    height = Column(Integer, nullable=False)
    width = Column(Integer, nullable=False)
    isProcessed = Column(Boolean, nullable=False, default=False, index=True)
    isClassDetected = Column(Boolean, nullable=False, default=False, index=True)
    isSubClassDetected = Column(Boolean, nullable=False, default=False, index=True)
    isSegmented = Column(Boolean, nullable=False, default=False, index=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    sample = relationship("Sample", back_populates="annotations")
    annotation_boxes = relationship("AnnotationBox", back_populates="annotation")

    __table_args__ = (
        Index('idx_annotations_id', 'id'),
        Index('idx_annotations_sample_id', 'sample_id'),
        Index('idx_annotations_isProcessed', 'isProcessed'),
        Index('idx_annotations_isClassDetected', 'isClassDetected'),
        Index('idx_annotations_isSubClassDetected', 'isSubClassDetected'),
        Index('idx_annotations_isSegmented', 'isSegmented'),
        Index('idx_annotations_created_at', 'created_at'),
    )


class AnnotationBox(Base):
    __tablename__ = "annotation_boxes"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    annotation_id = Column(String, ForeignKey("annotations.id"), nullable=False, index=True)
    boxes = Column(JSON, nullable=False)
    class_name = Column(String, nullable=True, index=True)
    class_confidence = Column(Float, nullable=True)
    sub_class = Column(String, nullable=True, index=True)
    sub_class_confidence = Column(Float, nullable=True)
    crop_path = Column(String, nullable=True)
    segmentation_polygon = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    annotation = relationship("Annotation", back_populates="annotation_boxes")
    wbc_sub_class_confidences = relationship(
        "WBCSubClassConfidences", back_populates="annotation_box", uselist=False
    )

    __table_args__ = (
        Index('idx_annotation_boxes_id', 'id'),
        Index('idx_annotation_boxes_annotation_id', 'annotation_id'),
        Index('idx_annotation_boxes_class_name', 'class_name'),
        Index('idx_annotation_boxes_sub_class', 'sub_class'),
    )


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
    neutrophil_confidence = Column(Float, nullable=True)
    lymphocyte_confidence = Column(Float, nullable=True)
    eosinophil_confidence = Column(Float, nullable=True)
    monocyte_confidence = Column(Float, nullable=True)
    basophil_confidence = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    annotation_box = relationship(
        "AnnotationBox", back_populates="wbc_sub_class_confidences"
    )

    __table_args__ = (
        Index('idx_wbc_sub_class_confidences_id', 'id'),
        Index('idx_wbc_sub_class_confidences_annotation_box_id', 'annotation_box_id'),
    )


class CellClasses(Base):
    __tablename__ = "cell_classes"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    name = Column(String, nullable=False, unique=True, index=True)
    colors = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    sub_classes = relationship("CellSubClasses", back_populates="cell_class")

    __table_args__ = (
        Index('idx_cell_classes_id', 'id'),
        Index('idx_cell_classes_name', 'name'),
    )


class CellSubClasses(Base):
    __tablename__ = "cell_sub_classes"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    class_id = Column(UUID(as_uuid=True), ForeignKey("cell_classes.id"), nullable=False, index=True)
    sub_class_name = Column(String, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    cell_class = relationship("CellClasses", back_populates="sub_classes")

    __table_args__ = (
        Index('idx_cell_sub_classes_id', 'id'),
        Index('idx_cell_sub_classes_class_id', 'class_id'),
        Index('idx_cell_sub_classes_sub_class_name', 'sub_class_name'),
    )


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
    test_id = Column(UUID(as_uuid=True), ForeignKey("tests.id"), nullable=False, index=True)
    assigned_at = Column(DateTime(timezone=True), server_default=func.now())

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
        Index('idx_test_assignments_id', 'id'),
        Index('idx_test_assignments_assigned_by_user_id', 'assigned_by_user_id'),
        Index('idx_test_assignments_assigned_to_user_id', 'assigned_to_user_id'),
        Index('idx_test_assignments_test_id', 'test_id'),
    )


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    action = Column(String, nullable=False, index=True)
    table_name = Column(String, nullable=False, index=True)
    user_ip_address = Column(String, nullable=False, server_default=text("inet_client_addr()"))
    old_data = Column(JSON, nullable=True)
    new_data = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)

    user = relationship("User", back_populates="audit_logs")

    __table_args__ = (
        Index('idx_audit_logs_id', 'id'),
        Index('idx_audit_logs_user_id', 'user_id'),
        Index('idx_audit_logs_action', 'action'),
        Index('idx_audit_logs_table_name', 'table_name'),
        Index('idx_audit_logs_created_at', 'created_at'),
    )


class Session(Base):
    __tablename__ = "sessions"

    id = Column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False, index=True)
    refresh_token_hash = Column(String, nullable=False)
    user_ip_address = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    last_activity_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    expires_at = Column(DateTime(timezone=True), nullable=False, index=True)
    revoked = Column(Boolean, nullable=False, default=False, index=True)

    user = relationship("User", back_populates="sessions")

    __table_args__ = (
        Index('idx_sessions_id', 'id'),
        Index('idx_sessions_user_id', 'user_id'),
        Index('idx_sessions_refresh_token_hash', 'refresh_token_hash'),
        Index('idx_sessions_expires_at', 'expires_at'),
        Index('idx_sessions_revoked', 'revoked'),
    )
